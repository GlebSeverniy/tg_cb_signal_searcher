from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import httpx
import pandas as pd
from bs4 import BeautifulSoup


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "extracted_channels.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "extracted_channels_with_subscribers.csv"
DEFAULT_DEBUG_HTML_DIR = PROJECT_ROOT / "data" / "debug" / "tgstat_channel_pages"
DEFAULT_USER_DATA_DIR = PROJECT_ROOT / "data" / "cache" / "tgstat_profile"

CHANNEL_COLUMN_CANDIDATES = [
    "channel_username",
    "username",
    "channel",
    "telegram_channel",
    "tg_channel",
    "handle",
]

ADDED_COLUMNS = [
    "tgstat_url",
    "subscribers",
    "subscribers_raw",
    "collected_at",
    "status",
    "error",
]

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:132.0) Gecko/20100101 Firefox/132.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
]

REQUEST_HEADERS = {
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

JSON_COUNT_RE = re.compile(
    r'"(?:participants_count|participantsCount|subscribers|subscribers_count|members_count)"\s*:\s*(\d{1,10})',
    re.IGNORECASE,
)
SUBSCRIBER_WORD_RE = r"(?:subscribers?|подписчиков|подписчики|подписчика|подписчик)"
NUMBER_RE = r"(?P<num>\d[\d\s\u00a0.,]*(?:\s*(?:[kKmMкКмМ]|млн))?)"
FORWARD_TEXT_RE = re.compile(rf"{NUMBER_RE}\s*{SUBSCRIBER_WORD_RE}", re.IGNORECASE)
REVERSE_TEXT_RE = re.compile(rf"{SUBSCRIBER_WORD_RE}\s*{NUMBER_RE}", re.IGNORECASE)
NEAR_NUMBER_RE = re.compile(r"\d[\d\s\u00a0.,]*(?:\s*(?:[kKmMкКмМ]|млн))?", re.IGNORECASE)
BLOCK_TOKENS = [
    ("captcha", "captcha keyword"),
    ("cloudflare", "cloudflare keyword"),
    ("access denied", "access denied keyword"),
    ("too many requests", "too many requests keyword"),
    ("rate limit", "rate limit keyword"),
    ("just a moment", "cloudflare just a moment text"),
    ("forbidden", "forbidden keyword"),
]


@dataclass(frozen=True)
class FetchResult:
    html: str | None
    status: str
    error: str
    http_status: int | None = None
    page_title: str = ""


@dataclass(frozen=True)
class ChannelResult:
    tgstat_url: str
    subscribers: int | None
    subscribers_raw: str | None
    collected_at: str
    status: str
    error: str
    html: str | None = None
    page_title: str = ""


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Telegram channel subscribers from TGStat.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay-min", type=float, default=3.0)
    parser.add_argument("--delay-max", type=float, default=9.0)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--use-playwright", action="store_true")
    parser.add_argument("--headless", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument(
        "--manual-wait",
        type=int,
        default=120,
        help="Seconds to keep a visible Playwright page open when TGStat shows a block/captcha page.",
    )
    parser.add_argument("--user-data-dir", type=Path, default=DEFAULT_USER_DATA_DIR)
    parser.add_argument("--debug-html-dir", type=Path, default=DEFAULT_DEBUG_HTML_DIR)
    parser.add_argument("--language", choices=["en", "ru"], default="en")
    return parser.parse_args()


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def normalize_channel_username(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    parsed = urlparse(text if "://" in text else f"https://{text}" if text.startswith("t.me/") else text)
    if parsed.netloc:
        host = parsed.netloc.lower()
        path_parts = [unquote(part) for part in parsed.path.split("/") if part]
        if "t.me" in host and path_parts:
            text = path_parts[0]
        elif "tgstat.ru" in host and "channel" in path_parts:
            idx = path_parts.index("channel")
            if idx + 1 < len(path_parts):
                text = path_parts[idx + 1]

    text = unquote(text).strip().strip("/")
    if text.startswith("@"):
        text = text[1:]
    if text.lower().startswith("https://t.me/") or text.lower().startswith("http://t.me/"):
        text = text.rsplit("/", 1)[-1].lstrip("@")
    if text.lower().startswith("t.me/"):
        text = text.split("/", 1)[1].lstrip("@")

    text = text.split("?", 1)[0].split("#", 1)[0].strip().strip("/")
    return text.lstrip("@")


def build_tgstat_url(username: str, language: str) -> str:
    prefix = "/en" if language == "en" else ""
    return f"https://tgstat.ru{prefix}/channel/{username}"


def detect_channel_column(df: pd.DataFrame) -> str:
    for column in CHANNEL_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(f"Cannot find channel column. Tried: {CHANNEL_COLUMN_CANDIDATES}. Available: {list(df.columns)}")


def parse_compact_number(raw: str) -> int:
    value = raw.replace("\u00a0", " ").strip().lower()
    value = re.sub(r"\s+", " ", value)
    multiplier = 1

    if re.search(r"(млн|m|м)\s*$", value, re.IGNORECASE):
        multiplier = 1_000_000
        value = re.sub(r"(млн|m|м)\s*$", "", value, flags=re.IGNORECASE).strip()
    elif re.search(r"(k|к)\s*$", value, re.IGNORECASE):
        multiplier = 1_000
        value = re.sub(r"(k|к)\s*$", "", value, flags=re.IGNORECASE).strip()

    compact = value.replace(" ", "")
    if not compact:
        raise ValueError(f"empty number: {raw!r}")

    separators = [char for char in compact if char in ".,"] 
    if multiplier > 1:
        compact = compact.replace(",", ".")
        if compact.count(".") > 1:
            parts = compact.split(".")
            compact = "".join(parts[:-1]) + "." + parts[-1]
        return int(float(compact) * multiplier)

    if separators:
        last_sep = max(compact.rfind("."), compact.rfind(","))
        suffix_len = len(compact) - last_sep - 1
        same_sep_count = compact.count(compact[last_sep])
        if suffix_len in {1, 2} and same_sep_count == 1:
            compact = compact.replace(",", ".")
            return int(float(compact))
        compact = compact.replace(".", "").replace(",", "")

    return int(compact)


def _clean_raw_number(raw: str) -> str:
    return re.sub(r"\s+", " ", raw.replace("\u00a0", " ")).strip()


def _valid_subscriber_count(value: int) -> bool:
    return 0 < value < 1_000_000_000


def _parse_number_match(raw: str) -> tuple[int, str] | None:
    raw = _clean_raw_number(raw)
    try:
        value = parse_compact_number(raw)
    except (TypeError, ValueError, OverflowError):
        return None
    if not _valid_subscriber_count(value):
        return None
    return value, raw


def _soup(html: str) -> BeautifulSoup:
    try:
        return BeautifulSoup(html, "lxml")
    except Exception:
        return BeautifulSoup(html, "html.parser")


def _extract_from_text(text: str) -> tuple[int | None, str | None]:
    for pattern in (FORWARD_TEXT_RE, REVERSE_TEXT_RE):
        best: tuple[int, str] | None = None
        for match in pattern.finditer(text):
            parsed = _parse_number_match(match.group("num"))
            if parsed and (best is None or parsed[0] > best[0]):
                best = parsed
        if best:
            return best
    return None, None


def _extract_from_dom_candidates(soup: BeautifulSoup) -> tuple[int | None, str | None]:
    candidates: list[tuple[int, int, str]] = []
    elements = soup.find_all(string=re.compile(SUBSCRIBER_WORD_RE, re.IGNORECASE))
    for node in elements:
        fragments = [str(node)]
        parent = getattr(node, "parent", None)
        if parent is not None:
            fragments.append(parent.get_text(" ", strip=True))
            if parent.parent is not None:
                fragments.append(parent.parent.get_text(" ", strip=True))
            for sibling in list(parent.previous_siblings)[:2] + list(parent.next_siblings)[:2]:
                get_text = getattr(sibling, "get_text", None)
                fragments.append(get_text(" ", strip=True) if get_text else str(sibling))

        for fragment in fragments:
            word_match = re.search(SUBSCRIBER_WORD_RE, fragment, re.IGNORECASE)
            if not word_match:
                continue
            for number_match in NEAR_NUMBER_RE.finditer(fragment):
                parsed = _parse_number_match(number_match.group(0))
                if not parsed:
                    continue
                distance = min(abs(number_match.start() - word_match.end()), abs(word_match.start() - number_match.end()))
                candidates.append((distance, parsed[0], parsed[1]))

    if not candidates:
        return None, None
    candidates.sort(key=lambda item: (item[0], -item[1]))
    _, value, raw = candidates[0]
    return value, raw


def _extract_from_meta(soup: BeautifulSoup) -> tuple[int | None, str | None]:
    fragments: list[str] = []
    if soup.title and soup.title.string:
        fragments.append(soup.title.string)
    for selector in [
        {"name": "description"},
        {"property": "og:title"},
        {"property": "og:description"},
    ]:
        tag = soup.find("meta", attrs=selector)
        if tag and tag.get("content"):
            fragments.append(str(tag["content"]))
    return _extract_from_text(" ".join(fragments))


def extract_subscribers(html: str) -> tuple[int | None, str | None, str | None]:
    for match in JSON_COUNT_RE.finditer(html):
        parsed = _parse_number_match(match.group(1))
        if parsed:
            return parsed[0], parsed[1], None

    soup = _soup(html)
    text = soup.get_text(" ", strip=True)
    value, raw = _extract_from_text(text)
    if value is not None:
        return value, raw, None

    value, raw = _extract_from_dom_candidates(soup)
    if value is not None:
        return value, raw, None

    value, raw = _extract_from_meta(soup)
    if value is not None:
        return value, raw, None

    return None, None, "subscribers count not found"


def extract_html_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def detect_block_reason(html: str) -> str | None:
    lower = html[:12000].lower()
    for token, reason in BLOCK_TOKENS:
        if token in lower:
            return reason
    return None


def classify_block(status_code: int | None, html: str | None) -> tuple[str, str] | None:
    reason = detect_block_reason(html or "") if html else None
    if status_code == 404:
        return "not_found", "HTTP 404"
    if status_code == 403:
        return ("captcha_or_blocked", f"HTTP 403: {reason}") if reason else ("forbidden", "HTTP 403")
    if status_code == 429:
        return "captcha_or_blocked", f"HTTP 429: {reason or 'too many requests'}"
    if reason:
        return "captcha_or_blocked", reason
    return None


def save_debug_html(args: argparse.Namespace, username: str, result: ChannelResult) -> None:
    if not result.html:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_username = re.sub(r"[^A-Za-z0-9._-]+", "_", username).strip("._") or "channel"
    args.debug_html_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.debug_html_dir / f"{safe_username}_{result.status}_{timestamp}.html"
    meta_path = html_path.with_suffix(".json")
    metadata = {
        "url": result.tgstat_url,
        "status": result.status,
        "error": result.error,
        "title": result.page_title,
        "html_length": len(result.html),
    }
    html_path.write_text(result.html, encoding="utf-8")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def fetch_httpx(client: httpx.Client, url: str, timeout: int, max_retries: int) -> FetchResult:
    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.get(url, headers={"User-Agent": random.choice(USER_AGENTS)}, timeout=timeout)
            html = response.text
            title = extract_html_title(html)
            block = classify_block(response.status_code, html)
            if block:
                status, error = block
                return FetchResult(html if response.status_code != 404 else None, status, error, response.status_code, title)
            if 500 <= response.status_code < 600:
                last_error = f"HTTP {response.status_code}"
                if attempt < max_retries:
                    time.sleep(min(10.0, 1.5 * attempt))
                    continue
                return FetchResult(html, "request_error", last_error, response.status_code, title)
            response.raise_for_status()
            return FetchResult(html, "ok", "", response.status_code, title)
        except httpx.HTTPError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if attempt < max_retries:
                time.sleep(min(10.0, 1.5 * attempt))
                continue
    return FetchResult(None, "request_error", last_error or "request failed")


class PlaywrightLoader:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.playwright: Any | None = None
        self.browser: Any | None = None
        self.context: Any | None = None
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        from playwright.sync_api import sync_playwright

        self.playwright = sync_playwright().start()
        kwargs: dict[str, Any] = {
            "headless": self.args.headless,
            "viewport": {"width": 1365, "height": 768},
            "locale": "ru-RU",
            "user_agent": random.choice(USER_AGENTS),
            "extra_http_headers": REQUEST_HEADERS,
            "args": ["--no-sandbox", "--disable-blink-features=AutomationControlled"],
        }
        if self.args.user_data_dir:
            self.args.user_data_dir.mkdir(parents=True, exist_ok=True)
            self.context = self.playwright.chromium.launch_persistent_context(
                user_data_dir=str(self.args.user_data_dir),
                **kwargs,
            )
        else:
            self.browser = self.playwright.chromium.launch(headless=self.args.headless, args=kwargs["args"])
            self.context = self.browser.new_context(
                user_agent=kwargs["user_agent"],
                locale=kwargs["locale"],
                extra_http_headers=kwargs["extra_http_headers"],
                viewport=kwargs["viewport"],
            )
        self.context.route("**/*", self._route_handler)
        self.started = True

    def _route_handler(self, route: Any) -> None:
        request = route.request
        url = request.url.lower()
        if request.resource_type in {"image", "font", "media"}:
            route.abort()
            return
        if any(token in url for token in ("gtag", "google-analytics", "mc.yandex", "metrika", "doubleclick", "facebook", "vk.com/rtrg")):
            route.abort()
            return
        route.continue_()

    def _read_page(self, page: Any) -> tuple[str, str]:
        html = page.content()
        try:
            title = page.title()
        except Exception:
            title = ""
        return html, title

    def _wait_for_manual_unblock(self, page: Any) -> tuple[str, str] | None:
        if self.args.headless or self.args.manual_wait <= 0:
            return None

        deadline = time.monotonic() + self.args.manual_wait
        print(
            f"TGStat shows a block/captcha page. Browser is open; solve/check it manually. "
            f"Waiting up to {self.args.manual_wait}s..."
        )
        while time.monotonic() < deadline:
            page.wait_for_timeout(2000)
            html, title = self._read_page(page)
            if classify_block(None, html) is None:
                return html, title
        return None

    def fetch(self, url: str, timeout: int) -> FetchResult:
        self.start()
        assert self.context is not None
        page = self.context.new_page()
        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)
            page.wait_for_timeout(random.randint(1000, 3000))
            html, title = self._read_page(page)
            status_code = response.status if response else None
            block = classify_block(status_code, html)
            if block:
                unblocked = self._wait_for_manual_unblock(page)
                if unblocked is not None:
                    html, title = unblocked
                    return FetchResult(html, "ok", "", status_code, title)
                status, error = block
                return FetchResult(html, status, error, status_code, title)
            return FetchResult(html, "ok", "", status_code, title)
        except Exception as exc:
            try:
                html = page.content()
                title = page.title()
            except Exception:
                html = None
                title = ""
            if html:
                block = classify_block(None, html)
                if block:
                    status, error = block
                    return FetchResult(html, status, f"{type(exc).__name__}: {error}", None, title)
            return FetchResult(None, "request_error", f"{type(exc).__name__}: {exc}")
        finally:
            page.close()

    def close(self) -> None:
        try:
            if self.context is not None:
                self.context.close()
            if self.browser is not None:
                self.browser.close()
        finally:
            if self.playwright is not None:
                self.playwright.stop()


def collect_channel(
    username: str,
    args: argparse.Namespace,
    client: httpx.Client,
    playwright: PlaywrightLoader | None,
) -> ChannelResult:
    url = build_tgstat_url(username, args.language)
    collected_at = datetime.now().isoformat(timespec="seconds")

    if args.use_playwright and playwright is not None:
        fetch = playwright.fetch(url, args.timeout)
    else:
        fetch = fetch_httpx(client, url, args.timeout, max(1, args.max_retries))

    if fetch.status != "ok":
        return ChannelResult(url, None, None, collected_at, fetch.status, fetch.error, fetch.html, fetch.page_title)

    if not fetch.html:
        return ChannelResult(url, None, None, collected_at, "parse_error", "empty HTML", None, fetch.page_title)

    subscribers, raw, error = extract_subscribers(fetch.html)
    if subscribers is None:
        return ChannelResult(url, None, raw, collected_at, "parse_error", error or "cannot parse subscribers", fetch.html, fetch.page_title)

    return ChannelResult(url, subscribers, raw, collected_at, "ok", "", None, fetch.page_title)


class IncrementalCsvWriter:
    def __init__(self, output_path: Path, fieldnames: list[str], append: bool) -> None:
        self.output_path = output_path
        self.fieldnames = fieldnames
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        write_header = not append or not output_path.exists() or output_path.stat().st_size == 0
        self.file = output_path.open(mode, encoding="utf-8-sig", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fieldnames)
        if write_header:
            self.writer.writeheader()
            self.file.flush()

    def write_row(self, row: dict[str, Any]) -> None:
        self.writer.writerow({column: row.get(column, "") for column in self.fieldnames})
        self.file.flush()

    def close(self) -> None:
        self.file.flush()
        self.file.close()


def load_completed_usernames(output_path: Path) -> set[str]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(output_path, usecols=lambda col: col in set(CHANNEL_COLUMN_CANDIDATES + ["status"]), dtype=str)
    except Exception:
        return set()
    if "status" not in df.columns:
        return set()
    channel_col = detect_channel_column(df)
    done = df[df["status"] == "ok"][channel_col].map(normalize_channel_username)
    return {username for username in done if username}


def output_has_resume_schema(output_path: Path) -> bool:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return False
    try:
        header = list(pd.read_csv(output_path, nrows=0).columns)
    except Exception:
        return False
    return "status" in header and any(column in header for column in CHANNEL_COLUMN_CANDIDATES)


def move_legacy_output(output_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    legacy_path = output_path.with_name(f"{output_path.stem}.legacy_{timestamp}{output_path.suffix}")
    counter = 1
    while legacy_path.exists():
        legacy_path = output_path.with_name(f"{output_path.stem}.legacy_{timestamp}_{counter}{output_path.suffix}")
        counter += 1
    output_path.replace(legacy_path)
    return legacy_path


def prepare_resume_output(output_path: Path, resume: bool) -> tuple[bool, set[str]]:
    if not resume:
        return False, set()
    if not output_path.exists() or output_path.stat().st_size == 0:
        return False, set()
    if output_has_resume_schema(output_path):
        return True, load_completed_usernames(output_path)

    legacy_path = move_legacy_output(output_path)
    print(f"Existing output has incompatible schema; moved it to {legacy_path}")
    return False, set()


def sleep_between(args: argparse.Namespace, blocked: bool) -> None:
    if blocked:
        delay = random.uniform(60, 180)
    else:
        delay = random.uniform(args.delay_min, args.delay_max)
    if delay > 0:
        time.sleep(delay)


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be greater than zero")
    if args.delay_min < 0 or args.delay_max < 0:
        raise SystemExit("--delay-min and --delay-max must be non-negative")
    if args.delay_min > args.delay_max:
        raise SystemExit("--delay-min must be <= --delay-max")
    if args.timeout <= 0:
        raise SystemExit("--timeout must be greater than zero")
    if args.max_retries <= 0:
        raise SystemExit("--max-retries must be greater than zero")


def run(args: argparse.Namespace) -> Counter:
    df = pd.read_csv(args.input)
    channel_col = detect_channel_column(df)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    fieldnames = list(df.columns) + [column for column in ADDED_COLUMNS if column not in df.columns]
    append_output, completed = prepare_resume_output(args.output, args.resume)
    writer = IncrementalCsvWriter(args.output, fieldnames, append=append_output)
    status_counts: Counter = Counter()
    cache: dict[str, ChannelResult] = {}
    consecutive_blocked = 0
    total = len(df)

    playwright = PlaywrightLoader(args) if args.use_playwright else None
    client = httpx.Client(http2=True, headers=REQUEST_HEADERS, follow_redirects=True)

    try:
        for idx, source_row in enumerate(df.to_dict("records"), start=1):
            username = normalize_channel_username(source_row.get(channel_col, ""))
            if not username:
                result = ChannelResult("", None, None, datetime.now().isoformat(timespec="seconds"), "parse_error", "empty channel username")
            elif username in completed:
                print(f"[{idx}/{total}] @{username}: skipped resume")
                continue
            elif username in cache:
                result = cache[username]
            else:
                result = collect_channel(username, args, client, playwright)
                cache[username] = result

            output_row = dict(source_row)
            output_row.update(
                {
                    "tgstat_url": result.tgstat_url,
                    "subscribers": result.subscribers if result.subscribers is not None else "",
                    "subscribers_raw": result.subscribers_raw or "",
                    "collected_at": result.collected_at,
                    "status": result.status,
                    "error": result.error,
                }
            )
            writer.write_row(output_row)
            status_counts[result.status] += 1

            if result.status == "ok":
                print(f"[{idx}/{total}] @{username}: {result.subscribers}")
                consecutive_blocked = 0
            else:
                print(f"[{idx}/{total}] @{username}: status={result.status} error={result.error}")
                if result.html and result.status in {"captcha_or_blocked", "forbidden", "parse_error"}:
                    save_debug_html(args, username or "unknown", result)
                if result.status in {"captcha_or_blocked", "forbidden"}:
                    consecutive_blocked += 1
                else:
                    consecutive_blocked = 0

            if consecutive_blocked >= 5:
                print("Stopping after 5 blocked/forbidden channels in a row. Progress is saved.")
                break

            if idx < total:
                sleep_between(args, result.status in {"captcha_or_blocked", "forbidden"})
    except KeyboardInterrupt:
        print("Interrupted. Progress already flushed to CSV.")
    finally:
        writer.close()
        client.close()
        if playwright is not None:
            playwright.close()

    return status_counts


def main() -> None:
    configure_stdout()
    args = parse_args()
    validate_args(args)
    counts = run(args)
    for status in ["ok", "not_found", "forbidden", "captcha_or_blocked", "parse_error", "request_error"]:
        print(f"Summary {status}: {counts.get(status, 0)}")


if __name__ == "__main__":
    main()
