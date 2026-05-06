import argparse
from pathlib import Path

import duckdb


DEFAULT_INPUT_FILE = Path("data/clean/extracted_messages.parquet")
DEFAULT_OUTPUT_DIR = Path("outputs/post_time_stats")


def interval_sql(hours: int) -> str:
    sign = "+" if hours >= 0 else "-"
    return f"{sign} INTERVAL {abs(hours)} HOUR"


def build_base_cte(input_file: Path, utc_offset_hours: int, only_messages: bool) -> str:
    message_filter = "AND message_type = 'message'" if only_messages else ""
    local_shift = interval_sql(utc_offset_hours)
    parquet_path = input_file.as_posix().replace("'", "''")

    return f"""
WITH posts AS (
    SELECT
        channel_id,
        channel_name,
        CAST(post_date AS TIMESTAMP) {local_shift} AS local_post_time
    FROM read_parquet('{parquet_path}')
    WHERE post_date IS NOT NULL
      {message_filter}
),
prepared AS (
    SELECT
        channel_id,
        channel_name,
        local_post_time,
        EXTRACT(hour FROM local_post_time)::INTEGER AS post_hour,
        EXTRACT(dow FROM local_post_time)::INTEGER AS weekday_num,
        CASE EXTRACT(dow FROM local_post_time)::INTEGER
            WHEN 0 THEN 'Sunday'
            WHEN 1 THEN 'Monday'
            WHEN 2 THEN 'Tuesday'
            WHEN 3 THEN 'Wednesday'
            WHEN 4 THEN 'Thursday'
            WHEN 5 THEN 'Friday'
            WHEN 6 THEN 'Saturday'
        END AS weekday_name
    FROM posts
    WHERE local_post_time IS NOT NULL
)
"""


def write_query(con: duckdb.DuckDBPyConnection, query: str, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_file.as_posix().replace("'", "''")
    con.execute(
        f"""
COPY (
{query}
) TO '{csv_path}'
WITH (HEADER, DELIMITER ',')
"""
    )
    add_utf8_bom(output_file)


def add_utf8_bom(output_file: Path):
    data = output_file.read_bytes()
    if not data.startswith(b"\xef\xbb\xbf"):
        output_file.write_bytes(b"\xef\xbb\xbf" + data)


def analyze_post_times(
    input_file: Path,
    output_dir: Path,
    utc_offset_hours: int,
    only_messages: bool,
):
    if not input_file.exists():
        raise FileNotFoundError(f"Input parquet file not found: {input_file}")

    output_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")
    base_cte = build_base_cte(input_file, utc_offset_hours, only_messages)

    by_hour_query = f"""
{base_cte}
SELECT
    post_hour,
    printf('%02d:00-%02d:59', post_hour, post_hour) AS hour_interval,
    COUNT(*) AS posts_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS posts_pct
FROM prepared
GROUP BY post_hour
ORDER BY posts_count DESC, post_hour
"""

    by_weekday_hour_query = f"""
{base_cte}
SELECT
    weekday_num,
    weekday_name,
    post_hour,
    printf('%02d:00-%02d:59', post_hour, post_hour) AS hour_interval,
    COUNT(*) AS posts_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS posts_pct
FROM prepared
GROUP BY weekday_num, weekday_name, post_hour
ORDER BY posts_count DESC, weekday_num, post_hour
"""

    channel_peak_hour_query = f"""
{base_cte},
channel_hour AS (
    SELECT
        channel_id,
        channel_name,
        post_hour,
        COUNT(*) AS posts_count
    FROM prepared
    GROUP BY channel_id, channel_name, post_hour
),
channel_totals AS (
    SELECT
        channel_id,
        channel_name,
        SUM(posts_count) AS channel_posts_count
    FROM channel_hour
    GROUP BY channel_id, channel_name
),
ranked AS (
    SELECT
        ch.channel_id,
        ch.channel_name,
        ch.post_hour,
        printf('%02d:00-%02d:59', ch.post_hour, ch.post_hour) AS peak_hour_interval,
        ch.posts_count AS peak_hour_posts_count,
        ct.channel_posts_count,
        ROUND(ch.posts_count * 100.0 / ct.channel_posts_count, 2) AS peak_hour_posts_pct,
        ROW_NUMBER() OVER (
            PARTITION BY ch.channel_id, ch.channel_name
            ORDER BY ch.posts_count DESC, ch.post_hour
        ) AS rn
    FROM channel_hour ch
    JOIN channel_totals ct
      ON ch.channel_id = ct.channel_id
     AND ch.channel_name = ct.channel_name
)
SELECT
    channel_name,
    channel_id,
    peak_hour_interval,
    peak_hour_posts_count,
    channel_posts_count,
    peak_hour_posts_pct
FROM ranked
WHERE rn = 1
ORDER BY channel_posts_count DESC, channel_name
"""

    summary_query = f"""
{base_cte}
SELECT
    COUNT(*) AS posts_count,
    MIN(local_post_time) AS earliest_post_time,
    MAX(local_post_time) AS latest_post_time,
    COUNT(DISTINCT channel_id) AS channels_count
FROM prepared
"""

    output_files = {
        "by_hour": output_dir / "posts_by_hour.csv",
        "by_weekday_hour": output_dir / "posts_by_weekday_hour.csv",
        "channel_peak_hour": output_dir / "channel_peak_hour.csv",
        "summary": output_dir / "post_time_summary.csv",
    }

    write_query(con, by_hour_query, output_files["by_hour"])
    write_query(con, by_weekday_hour_query, output_files["by_weekday_hour"])
    write_query(con, channel_peak_hour_query, output_files["channel_peak_hour"])
    write_query(con, summary_query, output_files["summary"])

    top_hour = con.execute(by_hour_query).fetchone()
    summary = con.execute(summary_query).fetchone()

    print(f"Input: {input_file}")
    print(f"UTC offset: {utc_offset_hours:+d} hours")
    print(f"Only message_type='message': {only_messages}")
    print(f"Posts analyzed: {summary[0]}")
    print(f"Channels analyzed: {summary[3]}")
    print(f"Date range: {summary[1]} - {summary[2]}")
    if top_hour:
        print(
            "Most frequent posting time: "
            f"{top_hour[1]} ({top_hour[2]} posts, {top_hour[3]}%)"
        )
    print(f"Saved: {output_files['by_hour']}")
    print(f"Saved: {output_files['by_weekday_hour']}")
    print(f"Saved: {output_files['channel_peak_hour']}")
    print(f"Saved: {output_files['summary']}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze what local time Telegram posts are published most often. "
            "Uses DuckDB SQL over parquet, so the whole file is not loaded into pandas."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_FILE,
        help=f"Path to extracted messages parquet. Default: {DEFAULT_INPUT_FILE}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV reports. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--utc-offset-hours",
        type=int,
        default=3,
        help="Hour offset from UTC for local post time. Default: 3 for Moscow time.",
    )
    parser.add_argument(
        "--include-service-posts",
        action="store_true",
        help="Include non-message rows. By default only message_type='message' is used.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    analyze_post_times(
        input_file=args.input,
        output_dir=args.output_dir,
        utc_offset_hours=args.utc_offset_hours,
        only_messages=not args.include_service_posts,
    )


if __name__ == "__main__":
    main()
