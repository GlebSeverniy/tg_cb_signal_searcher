r"""
tgstat_views_collector.py

Collect TGStat post view dynamics into long-format CSV.

Input columns:
    channel_username, post_id

Output columns are fixed by RESULT_COLUMNS.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import random
import re
import time
from collections import Counter, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "clean" / "extracted_messages.parquet"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "clean" / "tgstat_views.csv"
DEFAULT_HTML_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "tgstat_html"

RESULT_COLUMNS = [
    "channel_username",
    "post_id",
    "tgstat_url",
    "group_by",
    "x",
    "views_delta",
    "views_cumulative",
    "collected_at",
    "status",
    "error",
]

CHART_VARIABLES = {
    "chartData10min": "10min",
    "chartDataHour": "hour",
    "chartDataDay": "day",
}

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

FAST_CHART_RE = re.compile(
    r"\bvar\s+(chartData10min|chartDataHour|chartDataDay)\s*=\s*JSON\.parse\s*\(\s*'(\[(?:[^'\\]|\\.)*\])'\s*\)",
    re.DOTALL,
)
ANY_CHART_VAR_RE = re.compile(
    r"\bvar\s+(chartData10min|chartDataHour|chartDataDay)\s*=\s*JSON\.parse\s*\(",
    re.DOTALL,
)


class NoChartDataError(Exception):
    pass


class ChartParseError(Exception):
    pass


@dataclass(frozen=True)
class PostTask:
    channel_username: str
    post_id: str
    tgstat_url: str


@dataclass
class FetchResult:
    html: str | None
    status: str
    error: str
    collected_at: str
    block_signal: bool = False
    detected_by: str = ""
    page_title: str = ""


@dataclass
class PostResult:
    rows: list[dict]
    post_count: int = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect TGStat post view dynamics.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--html-cache-dir", type=Path, default=DEFAULT_HTML_CACHE_DIR)
    parser.add_argument("--debug-html-dir", type=Path, default=Path("data/debug/tgstat_html"))
    parser.add_argument("--use-playwright", action="store_true", help="Use Playwright as the primary loader.")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--min-concurrency", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-base-delay", type=float, default=5.0)
    parser.add_argument("--rate-limit-delay", type=float, default=90.0)
    parser.add_argument("--throttle-cooldown", type=float, default=60.0)
    parser.add_argument("--recovery-window", type=int, default=20)
    parser.add_argument("--recovery-step", type=int, default=1)
    parser.add_argument("--throttle-step", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--playwright-wait", type=float, default=0.7)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--channel-col", default="channel_username")
    parser.add_argument("--post-id-col", default="post_id")
    parser.add_argument("--tgstat-language", choices=["en", "ru"], default="en")
    parser.add_argument("--batch-posts", type=int, default=100)
    parser.add_argument("--save-html-cache", action="store_true")
    parser.add_argument("--save-extracted-json", type=Path, default=None)
    parser.add_argument("--jsonl-batch-posts", type=int, default=10000)
    parser.add_argument("--playwright-channel", default="chrome")
    parser.add_argument("--channel", default=None, help="Alias for --playwright-channel.")
    parser.add_argument("--user-data-dir", type=Path, default=None)
    parser.add_argument("--headless", type=_str_to_bool, nargs="?", const=True, default=False)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--playwright-first", action="store_true", help="Deprecated alias for --use-playwright.")
    parser.add_argument("--sleep", type=float, default=0.0, help="Deprecated; ignored.")
    parser.add_argument("--sleep-jitter", type=float, default=0.0, help="Deprecated; ignored.")
    parser.add_argument("--retries", type=int, default=None, help="Deprecated alias for --max-retries.")
    parser.add_argument("--parse-workers", type=int, default=0, help="Deprecated; parsing is in-process.")
    return parser.parse_args()


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError("Expected true/false")


def read_input_table(input_path: Path) -> pd.DataFrame:
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_path)
    if suffix == ".csv":
        return pd.read_csv(input_path)
    raise ValueError(f"Unsupported input format: {input_path.suffix}")


def validate_columns(df: pd.DataFrame, channel_col: str, post_id_col: str) -> None:
    missing = [col for col in (channel_col, post_id_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Available: {list(df.columns)}")


def normalize_channel_username(username: str) -> str:
    if pd.isna(username):
        raise ValueError("channel username is empty")
    value = str(username).strip()
    if value.startswith("@"):
        value = value[1:]
    if not value:
        raise ValueError("channel username is empty")
    return value


def _normalize_post_id(post_id: int | str | Any) -> str:
    if pd.isna(post_id):
        raise ValueError("post_id is empty")
    if isinstance(post_id, float) and post_id.is_integer():
        return str(int(post_id))
    value = str(post_id).strip()
    if value.endswith(".0") and value[:-2].isdigit():
        value = value[:-2]
    if not value:
        raise ValueError("post_id is empty")
    return value


def build_tgstat_stat_url(channel_username: str, post_id: int | str, language: str = "en") -> str:
    username = normalize_channel_username(channel_username)
    post = _normalize_post_id(post_id)
    prefix = "/en" if language == "en" else ""
    return f"https://tgstat.ru{prefix}/channel/@{username}/{quote(post, safe='')}/stat"


def safe_cache_filename(channel_username: str, post_id: int | str) -> str:
    username = normalize_channel_username(channel_username)
    post = _normalize_post_id(post_id)
    return re.sub(r"[^A-Za-z0-9._-]+", "_", f"{username}_{post}.html")


def html_cache_path(root: Path, channel_username: str, post_id: str) -> Path:
    username = normalize_channel_username(channel_username)
    shard = username[:2].lower() or "__"
    return root / shard / safe_cache_filename(username, post_id)


def detect_block_reason(html: str) -> str | None:
    if html_has_chart_data(html):
        return None
    lower = html[:8000].lower()
    checks = [
        ("captcha", "captcha keyword"),
        ("cloudflare", "cloudflare keyword"),
        ("too many requests", "too many requests keyword"),
        ("access denied", "access denied keyword"),
        ("rate limit", "rate limit keyword"),
        ("just a moment", "cloudflare just a moment text"),
        ("forbidden", "forbidden keyword"),
    ]
    for token, reason in checks:
        if token in lower:
            return reason
    return None


def looks_blocked_text(html: str) -> bool:
    return detect_block_reason(html) is not None


def cache_is_valid(html: str) -> bool:
    return html_has_chart_data(html)


def extract_html_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return re.sub(r"\s+", " ", match.group(1)).strip()


def load_completed_posts(output_path: Path) -> set[tuple[str, str]]:
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    completed: set[tuple[str, str]] = set()
    for chunk in pd.read_csv(output_path, usecols=["channel_username", "post_id", "status"], dtype=str, chunksize=200_000):
        chunk = chunk[chunk["status"] == "ok"]
        channels = chunk["channel_username"].map(lambda value: normalize_channel_username(value) if pd.notna(value) else "")
        posts = chunk["post_id"].map(_normalize_post_id)
        completed.update(zip(channels, posts))
    return completed


def make_post_tasks(df: pd.DataFrame, args: argparse.Namespace) -> list[PostTask]:
    tasks: list[PostTask] = []
    for record in df[[args.channel_col, args.post_id_col]].to_dict("records"):
        username = normalize_channel_username(record[args.channel_col])
        post_id = _normalize_post_id(record[args.post_id_col])
        tasks.append(PostTask(username, post_id, build_tgstat_stat_url(username, post_id, args.tgstat_language)))
    seen: set[tuple[str, str]] = set()
    unique: list[PostTask] = []
    for task in tasks:
        key = (task.channel_username, task.post_id)
        if key not in seen:
            unique.append(task)
            seen.add(key)
    return unique


def make_error_row(channel_username: str, post_id: int | str, tgstat_url: str, collected_at: str, status: str, error: str) -> dict:
    return {
        "channel_username": normalize_channel_username(channel_username) if channel_username else "",
        "post_id": _normalize_post_id(post_id) if str(post_id).strip() else "",
        "tgstat_url": tgstat_url,
        "group_by": "",
        "x": "",
        "views_delta": "",
        "views_cumulative": "",
        "collected_at": collected_at,
        "status": status,
        "error": error,
    }


def chart_data_to_rows(channel_username: str, post_id: int | str, tgstat_url: str, charts: dict[str, list[dict]], collected_at: str) -> list[dict]:
    rows: list[dict] = []
    for group_by in ("10min", "hour", "day"):
        cumulative = 0
        for point in charts.get(group_by, []):
            views_delta = int(point["y"])
            cumulative += views_delta
            rows.append(
                {
                    "channel_username": normalize_channel_username(channel_username),
                    "post_id": _normalize_post_id(post_id),
                    "tgstat_url": tgstat_url,
                    "group_by": group_by,
                    "x": point.get("x", ""),
                    "views_delta": views_delta,
                    "views_cumulative": cumulative,
                    "collected_at": collected_at,
                    "status": "ok",
                    "error": "",
                }
            )
    return rows


def html_has_chart_data(html: str) -> bool:
    return bool(ANY_CHART_VAR_RE.search(html))


def extract_chart_json(html: str, variable_name: str) -> list[dict]:
    pattern = re.compile(
        rf"\bvar\s+{re.escape(variable_name)}\s*=\s*JSON\.parse\s*\(\s*'(\[(?:[^'\\]|\\.)*\])'\s*\)",
        re.DOTALL,
    )
    match = pattern.search(html)
    if match:
        return _loads_chart_points(_decode_js_string(match.group(1)), variable_name)

    marker = re.compile(rf"\bvar\s+{re.escape(variable_name)}\s*=\s*JSON\.parse\s*\(", re.DOTALL).search(html)
    if not marker:
        raise NoChartDataError(f"{variable_name} not found")
    idx = marker.end()
    while idx < len(html) and html[idx].isspace():
        idx += 1
    if idx >= len(html) or html[idx] not in {"'", '"'}:
        raise ChartParseError(f"{variable_name}: JSON.parse argument is not a quoted JS string")
    raw, _ = _read_js_string_literal(html, idx)
    return _loads_chart_points(_decode_js_string(raw), variable_name)


def extract_all_chart_data(html: str) -> dict[str, list[dict]]:
    charts: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for match in FAST_CHART_RE.finditer(html):
        variable = match.group(1)
        charts[CHART_VARIABLES[variable]] = _loads_chart_points(_decode_js_string(match.group(2)), variable)
        seen.add(variable)
    if len(seen) < len(CHART_VARIABLES) and ANY_CHART_VAR_RE.search(html):
        for match in ANY_CHART_VAR_RE.finditer(html):
            variable = match.group(1)
            if variable not in seen:
                charts[CHART_VARIABLES[variable]] = extract_chart_json(html, variable)
                seen.add(variable)
    if not charts:
        raise NoChartDataError("No chart variables found")
    return charts


def _loads_chart_points(decoded_json: str, variable_name: str) -> list[dict]:
    try:
        points = json.loads(decoded_json)
    except json.JSONDecodeError as exc:
        raise ChartParseError(f"{variable_name}: cannot parse JSON: {exc}") from exc
    if not isinstance(points, list):
        raise ChartParseError(f"{variable_name}: expected list")
    normalized: list[dict] = []
    for idx, point in enumerate(points):
        if not isinstance(point, dict):
            raise ChartParseError(f"{variable_name}: point #{idx} is not an object")
        try:
            y = int(point.get("y", 0))
        except (TypeError, ValueError) as exc:
            raise ChartParseError(f"{variable_name}: point #{idx} has non-int y") from exc
        item = dict(point)
        item["y"] = y
        normalized.append(item)
    return normalized


def _read_js_string_literal(text: str, quote_index: int) -> tuple[str, int]:
    quote_char = text[quote_index]
    chars: list[str] = []
    idx = quote_index + 1
    escaped = False
    while idx < len(text):
        char = text[idx]
        if escaped:
            chars.append("\\" + char)
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == quote_char:
            return "".join(chars), idx + 1
        else:
            chars.append(char)
        idx += 1
    raise ValueError("unterminated JS string literal")


def _decode_js_string(raw: str) -> str:
    decoded: list[str] = []
    idx = 0
    while idx < len(raw):
        char = raw[idx]
        if char != "\\":
            decoded.append(char)
            idx += 1
            continue
        idx += 1
        if idx >= len(raw):
            decoded.append("\\")
            break
        escape = raw[idx]
        if escape in {"'", '"', "\\", "/"}:
            decoded.append(escape)
            idx += 1
        elif escape == "n":
            decoded.append("\n")
            idx += 1
        elif escape == "r":
            decoded.append("\r")
            idx += 1
        elif escape == "t":
            decoded.append("\t")
            idx += 1
        elif escape == "u" and idx + 4 < len(raw):
            decoded.append(chr(int(raw[idx + 1 : idx + 5], 16)))
            idx += 5
        elif escape == "x" and idx + 2 < len(raw):
            decoded.append(chr(int(raw[idx + 1 : idx + 3], 16)))
            idx += 3
        else:
            decoded.append(escape)
            idx += 1
    return "".join(decoded)


class AdaptiveConcurrencyController:
    def __init__(self, args: argparse.Namespace) -> None:
        self.max_concurrency = args.concurrency
        self.min_concurrency = min(args.min_concurrency, args.concurrency)
        self.current_limit = args.concurrency
        self.throttle_cooldown = args.throttle_cooldown
        self.rate_limit_delay = args.rate_limit_delay
        self.recovery_window = args.recovery_window
        self.recovery_step = max(1, args.recovery_step)
        self.throttle_step = min(max(args.throttle_step, 0.05), 0.8)
        self.active = 0
        self.success_streak = 0
        self.last_throttle_at = 0.0
        self.pause_until = 0.0
        self.recent_blocks: deque[float] = deque()
        self.condition = asyncio.Condition()
        self.lock = asyncio.Lock()

    async def wait_if_paused(self) -> None:
        while True:
            wait_seconds = max(0.0, self.pause_until - time.monotonic())
            if wait_seconds <= 0:
                return
            await asyncio.sleep(min(wait_seconds, 1.0))

    async def acquire(self) -> None:
        while True:
            await self.wait_if_paused()
            async with self.condition:
                if self.active < self.current_limit and time.monotonic() >= self.pause_until:
                    self.active += 1
                    return
                await self.condition.wait()

    async def release(self) -> None:
        async with self.condition:
            self.active = max(0, self.active - 1)
            self.condition.notify_all()

    async def record_success(self) -> None:
        async with self.lock:
            self.success_streak += 1
            if self.current_limit < self.max_concurrency and self.success_streak >= self.recovery_window:
                old = self.current_limit
                self.current_limit = min(self.max_concurrency, self.current_limit + self.recovery_step)
                self.success_streak = 0
                logging.info("Recovery: concurrency %s -> %s", old, self.current_limit)
        async with self.condition:
            self.condition.notify_all()

    async def record_block(self, reason: str) -> bool:
        async with self.lock:
            now = time.monotonic()
            self.success_streak = 0
            self.recent_blocks.append(now)
            while self.recent_blocks and now - self.recent_blocks[0] > 30:
                self.recent_blocks.popleft()
            block_threshold = min(3, max(1, self.current_limit))
            if len(self.recent_blocks) < block_threshold:
                return False

            pause_seconds = self.pause_until - now
            if pause_seconds > 0:
                return True

            can_reduce = now - self.last_throttle_at >= self.throttle_cooldown and self.current_limit > self.min_concurrency
            old = self.current_limit
            if can_reduce:
                drop = max(1, math.ceil(old * self.throttle_step))
                self.current_limit = max(self.min_concurrency, old - drop)
                self.last_throttle_at = now
            self.pause_until = now + self.rate_limit_delay
            self.recent_blocks.clear()
            if can_reduce:
                logging.warning(
                    "Rate limit detected: pausing new tasks for %.0f sec, concurrency %s -> %s (%s)",
                    self.rate_limit_delay,
                    old,
                    self.current_limit,
                    reason,
                )
            else:
                logging.warning(
                    "Rate limit detected: pausing new tasks for %.0f sec, concurrency remains %s (%s)",
                    self.rate_limit_delay,
                    self.current_limit,
                    reason,
                )
        async with self.condition:
            self.condition.notify_all()
        return True


def retry_delay(base_delay: float, attempt: int) -> float:
    return base_delay * (2 ** max(0, attempt - 1)) * random.uniform(0.8, 1.2)


class PlaywrightLoader:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.playwright: Any | None = None
        self.browser: Any | None = None
        self.context: Any | None = None
        self.pages: asyncio.Queue[Any] = asyncio.Queue()
        self.started = False
        self.start_lock = asyncio.Lock()
        self.warned_channel = False

    async def start(self) -> None:
        if self.started:
            return
        async with self.start_lock:
            if self.started:
                return
            from playwright.async_api import async_playwright

            self.playwright = await async_playwright().start()
            if self.args.user_data_dir:
                self.args.user_data_dir.mkdir(parents=True, exist_ok=True)
                kwargs: dict[str, Any] = {
                    "user_data_dir": str(self.args.user_data_dir),
                    "headless": self.args.headless,
                    "viewport": {"width": 1365, "height": 768},
                    "locale": "ru-RU",
                    "user_agent": USER_AGENTS[0],
                    "extra_http_headers": REQUEST_HEADERS,
                    "args": ["--no-sandbox", "--disable-blink-features=AutomationControlled"],
                }
                if self.args.playwright_channel:
                    kwargs["channel"] = self.args.playwright_channel
                try:
                    self.context = await self.playwright.chromium.launch_persistent_context(**kwargs)
                except Exception as exc:
                    if self.args.playwright_channel:
                        if not self.warned_channel:
                            logging.warning("Cannot launch persistent channel=%s: %s. Falling back to bundled Chromium.", self.args.playwright_channel, exc)
                            self.warned_channel = True
                        kwargs.pop("channel", None)
                        self.context = await self.playwright.chromium.launch_persistent_context(**kwargs)
                    else:
                        raise
                self.browser = None
                logging.info("Using persistent Playwright profile: %s", self.args.user_data_dir)
            else:
                kwargs = {
                    "headless": self.args.headless,
                    "args": ["--no-sandbox", "--disable-blink-features=AutomationControlled"],
                }
                if self.args.playwright_channel:
                    kwargs["channel"] = self.args.playwright_channel
                try:
                    self.browser = await self.playwright.chromium.launch(**kwargs)
                except Exception as exc:
                    if self.args.playwright_channel:
                        if not self.warned_channel:
                            logging.warning("Cannot launch channel=%s: %s. Falling back to bundled Chromium.", self.args.playwright_channel, exc)
                            self.warned_channel = True
                        kwargs.pop("channel", None)
                        self.browser = await self.playwright.chromium.launch(**kwargs)
                    else:
                        raise
                self.context = await self.browser.new_context(
                    user_agent=random.choice(USER_AGENTS),
                    locale="ru-RU",
                    extra_http_headers=REQUEST_HEADERS,
                    viewport={"width": 1365, "height": 768},
                )
            await self.context.route("**/*", self._route_handler)
            for _ in range(self.args.concurrency):
                await self.pages.put(await self.context.new_page())
            self.started = True

    async def _route_handler(self, route: Any) -> None:
        req = route.request
        url = req.url.lower()
        if req.resource_type in {"image", "font", "stylesheet", "media"}:
            await route.abort()
            return
        if any(token in url for token in ("gtag", "google-analytics", "mc.yandex", "metrika", "doubleclick", "facebook", "vk.com/rtrg")):
            await route.abort()
            return
        await route.continue_()

    async def fetch(self, task: PostTask) -> FetchResult:
        await self.start()
        page = await self.pages.get()
        replace_page = False
        try:
            await page.goto(task.tgstat_url, wait_until="domcontentloaded", timeout=self.args.timeout * 1000)
            if self.args.playwright_wait > 0:
                await page.wait_for_timeout(int(self.args.playwright_wait * 1000))
            html = await page.content()
            try:
                page_title = await page.title()
            except Exception:
                page_title = ""
            status = "ok"
            error = ""
            detected_by = detect_block_reason(html) or ""
            block = bool(detected_by)
            if block:
                status = "forbidden"
                error = "Blocked/captcha/rate-limit page"
            return FetchResult(html, status, error, datetime.now().isoformat(timespec="seconds"), block, detected_by, page_title)
        except Exception as exc:
            replace_page = True
            try:
                await page.close()
            except Exception:
                pass
            return FetchResult(None, "request_error", f"{type(exc).__name__}: {exc}", datetime.now().isoformat(timespec="seconds"))
        finally:
            if replace_page and self.context is not None:
                try:
                    await self.pages.put(await self.context.new_page())
                except Exception:
                    pass
            elif not page.is_closed():
                await self.pages.put(page)

    async def close(self) -> None:
        try:
            while not self.pages.empty():
                page = await self.pages.get()
                try:
                    await page.close()
                except Exception:
                    pass
            if self.context is not None:
                try:
                    await self.context.close()
                except Exception:
                    pass
            if self.browser is not None:
                try:
                    await self.browser.close()
                except Exception:
                    pass
        finally:
            if self.playwright is not None:
                try:
                    await self.playwright.stop()
                except Exception:
                    pass


async def create_async_client(timeout: int) -> httpx.AsyncClient:
    return httpx.AsyncClient(http2=True, headers=REQUEST_HEADERS, timeout=httpx.Timeout(timeout), follow_redirects=True)


async def fetch_httpx(client: httpx.AsyncClient, task: PostTask, args: argparse.Namespace) -> FetchResult:
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        resp = await client.get(task.tgstat_url, headers=headers)
        html = resp.text
        collected = datetime.now().isoformat(timespec="seconds")
        if resp.status_code == 404:
            return FetchResult(None, "not_found", "HTTP 404", collected)
        detected_by = detect_block_reason(html)
        if resp.status_code in {403, 429} and not html_has_chart_data(html):
            detected_by = detected_by or f"HTTP {resp.status_code}"
        if detected_by:
            return FetchResult(html, "forbidden", f"HTTP {resp.status_code}: blocked or rate-limited", collected, True, detected_by, extract_html_title(html))
        if 500 <= resp.status_code < 600:
            return FetchResult(html, "request_error", f"HTTP {resp.status_code}", collected, False, "", extract_html_title(html))
        resp.raise_for_status()
        return FetchResult(html, "ok", "", collected, False, "", extract_html_title(html))
    except httpx.HTTPError as exc:
        return FetchResult(None, "request_error", f"{type(exc).__name__}: {exc}", datetime.now().isoformat(timespec="seconds"))


def parse_post_from_html(task: PostTask, html: str, collected_at: str) -> PostResult:
    if not html_has_chart_data(html):
        return PostResult([make_error_row(task.channel_username, task.post_id, task.tgstat_url, collected_at, "no_chart_data", "HTML does not contain chartData variables")])
    try:
        charts = extract_all_chart_data(html)
        rows = chart_data_to_rows(task.channel_username, task.post_id, task.tgstat_url, charts, collected_at)
        counts = [len(charts.get(group, [])) for group in ("10min", "hour", "day")]
        logging.info("Success: %s/%s extracted %s/%s/%s points", task.channel_username, task.post_id, *counts)
        return PostResult(rows or [make_error_row(task.channel_username, task.post_id, task.tgstat_url, collected_at, "no_chart_data", "Chart variables are empty")])
    except NoChartDataError as exc:
        return PostResult([make_error_row(task.channel_username, task.post_id, task.tgstat_url, collected_at, "no_chart_data", str(exc))])
    except ChartParseError as exc:
        return PostResult([make_error_row(task.channel_username, task.post_id, task.tgstat_url, collected_at, "parse_error", str(exc))])


async def parse_post_result_with_debug(
    args: argparse.Namespace,
    task: PostTask,
    html: str,
    fetch: FetchResult,
) -> PostResult:
    result = parse_post_from_html(task, html, fetch.collected_at)
    first = result.rows[0] if result.rows else {}
    if first.get("status") == "no_chart_data":
        detected_by = detect_block_reason(html)
        if not detected_by:
            await save_debug_html(
                args,
                task,
                html,
                "no_chart_data",
                "no_chart_data",
                first.get("error", "HTML does not contain chartData variables"),
                "chartData variables not found",
                fetch.page_title,
            )
    return result


async def maybe_read_valid_cache(args: argparse.Namespace, task: PostTask) -> str | None:
    path = html_cache_path(args.html_cache_dir, task.channel_username, task.post_id)
    if not path.exists():
        return None
    html = await asyncio.to_thread(path.read_text, "utf-8", errors="replace")
    if cache_is_valid(html):
        return html
    return None


async def maybe_save_html(args: argparse.Namespace, task: PostTask, html: str) -> None:
    if not args.save_html_cache:
        return
    path = html_cache_path(args.html_cache_dir, task.channel_username, task.post_id)
    if not html_has_chart_data(html):
        suffix = ".forbidden.html" if looks_blocked_text(html) else ".error.html"
        path = path.with_suffix(suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_text, html, "utf-8")


def debug_safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "value"


async def save_debug_html(
    args: argparse.Namespace,
    task: PostTask,
    html: str,
    kind: str,
    status: str,
    reason: str,
    detected_by: str,
    page_title: str = "",
) -> None:
    if not html:
        return
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    channel = debug_safe_name(normalize_channel_username(task.channel_username))
    post_id = debug_safe_name(_normalize_post_id(task.post_id))
    prefix = "blocked" if kind == "blocked" else "no_chart_data"
    args.debug_html_dir.mkdir(parents=True, exist_ok=True)
    html_path = args.debug_html_dir / f"{prefix}_{channel}_{post_id}_{timestamp}.html"
    meta_path = html_path.with_suffix(".json")
    metadata = {
        "url": task.tgstat_url,
        "status": status,
        "reason": reason,
        "detected_by": detected_by,
        "html_length": len(html),
        "page_title": page_title,
    }
    await asyncio.to_thread(html_path.write_text, html, "utf-8")
    await asyncio.to_thread(meta_path.write_text, json.dumps(metadata, ensure_ascii=False, indent=2), "utf-8")
    logging.info("DEBUG HTML saved: %s", html_path)


async def process_task(
    task: PostTask,
    args: argparse.Namespace,
    controller: AdaptiveConcurrencyController,
    client: httpx.AsyncClient | None,
    playwright: PlaywrightLoader | None,
) -> PostResult:
    cached = await maybe_read_valid_cache(args, task)
    if cached is not None:
        fetch = FetchResult(cached, "ok", "", datetime.now().isoformat(timespec="seconds"))
        return await parse_post_result_with_debug(args, task, cached, fetch)

    await controller.acquire()
    try:
        last_block: FetchResult | None = None
        for attempt in range(1, args.max_retries + 1):
            logging.info("Processing post: %s/%s", task.channel_username, task.post_id)
            fetch = await (playwright.fetch(task) if args.use_playwright else fetch_httpx(client, task, args))
            if fetch.html:
                await maybe_save_html(args, task, fetch.html)

            if fetch.status == "not_found":
                return PostResult([make_error_row(task.channel_username, task.post_id, task.tgstat_url, fetch.collected_at, "not_found", fetch.error)])

            if fetch.status == "ok" and fetch.html:
                result = await parse_post_result_with_debug(args, task, fetch.html, fetch)
                first_status = result.rows[0]["status"] if result.rows else "parse_error"
                if first_status == "ok":
                    await controller.record_success()
                    return result
                if first_status == "no_chart_data" and looks_blocked_text(fetch.html):
                    fetch.block_signal = True
                    fetch.status = "forbidden"
                    fetch.error = "Blocked/captcha/rate-limit page without chartData"
                    fetch.detected_by = detect_block_reason(fetch.html) or fetch.detected_by
                    last_block = fetch
                elif first_status in {"parse_error"}:
                    return result
                elif attempt >= args.max_retries:
                    return result
            if fetch.block_signal:
                last_block = fetch
                if fetch.html:
                    await save_debug_html(
                        args,
                        task,
                        fetch.html,
                        "blocked",
                        "forbidden",
                        fetch.error or "Blocked/captcha/rate-limit page",
                        fetch.detected_by or detect_block_reason(fetch.html) or "blocked signal",
                        fetch.page_title,
                    )
                await controller.record_block(fetch.error or "blocked/captcha/rate-limit")
                if attempt >= args.max_retries:
                    return PostResult([
                        make_error_row(
                            task.channel_username,
                            task.post_id,
                            task.tgstat_url,
                            fetch.collected_at,
                            "forbidden",
                            fetch.error or "Blocked/captcha/rate-limit page",
                        )
                    ])
                delay = args.rate_limit_delay * random.uniform(0.8, 1.2)
                logging.info(
                    "Retry %s/%s after rate-limit pause: %s/%s reason=%s next_delay=%.1fs",
                    attempt,
                    args.max_retries,
                    task.channel_username,
                    task.post_id,
                    fetch.error,
                    delay,
                )
                await asyncio.sleep(delay)
                continue
            if fetch.status in {"request_error", "forbidden"} and attempt >= args.max_retries:
                return PostResult([make_error_row(task.channel_username, task.post_id, task.tgstat_url, fetch.collected_at, fetch.status, fetch.error)])

            delay = retry_delay(args.retry_base_delay, attempt)
            logging.info("Retry %s/%s: %s/%s reason=%s next_delay=%.1fs", attempt, args.max_retries, task.channel_username, task.post_id, fetch.error, delay)
            await asyncio.sleep(delay)

        now = datetime.now().isoformat(timespec="seconds")
        if last_block is not None:
            return PostResult([
                make_error_row(
                    task.channel_username,
                    task.post_id,
                    task.tgstat_url,
                    now,
                    "forbidden",
                    last_block.error or "Blocked/captcha/rate-limit page after retries",
                )
            ])
        return PostResult([make_error_row(task.channel_username, task.post_id, task.tgstat_url, now, "request_error", "Retries exhausted")])
    finally:
        await controller.release()


class IncrementalCsvWriter:
    def __init__(self, output_path: Path) -> None:
        self.output_path = output_path
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        exists = output_path.exists() and output_path.stat().st_size > 0
        self.file = output_path.open("a", encoding="utf-8-sig", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=RESULT_COLUMNS)
        if not exists:
            self.writer.writeheader()

    def write_rows(self, rows: list[dict]) -> None:
        for row in rows:
            self.writer.writerow({col: row.get(col, "") for col in RESULT_COLUMNS})

    def close(self) -> None:
        self.file.flush()
        self.file.close()


def prepare_tasks(args: argparse.Namespace) -> list[PostTask]:
    df = read_input_table(args.input)
    logging.info("Loaded %s rows from %s", len(df), args.input)
    validate_columns(df, args.channel_col, args.post_id_col)
    if args.limit is not None:
        df = df.head(args.limit)
    tasks = make_post_tasks(df, args)
    if args.resume:
        completed = load_completed_posts(args.output)
        before = len(tasks)
        tasks = [task for task in tasks if (task.channel_username, task.post_id) not in completed]
        logging.info("Resume: skipped %s already completed posts from %s", before - len(tasks), args.output)
    return tasks


def validate_args(args: argparse.Namespace) -> None:
    if args.playwright_first:
        logging.warning("--playwright-first is deprecated; treating it as --use-playwright.")
        args.use_playwright = True
    if args.channel:
        args.playwright_channel = args.channel
    if args.sleep or args.sleep_jitter:
        logging.warning("--sleep and --sleep-jitter are deprecated and ignored.")
    if args.retries is not None:
        logging.warning("--retries is deprecated; using it as --max-retries.")
        args.max_retries = args.retries
    if args.parse_workers:
        logging.warning("--parse-workers is deprecated on Windows with Playwright; parsing will remain in-process.")
        args.parse_workers = 0
    if args.concurrency < 1 or args.concurrency > 50:
        raise SystemExit("--concurrency must be in range 1..50")
    if args.min_concurrency < 1:
        raise SystemExit("--min-concurrency must be >= 1")
    if args.rate_limit_delay <= 0:
        raise SystemExit("--rate-limit-delay must be > 0")
    args.min_concurrency = min(args.min_concurrency, args.concurrency)
    if args.batch_posts < 1:
        raise SystemExit("--batch-posts must be >= 1")


async def run_collector(args: argparse.Namespace) -> Counter:
    tasks = prepare_tasks(args)
    total = len(tasks)
    if not tasks:
        logging.info("No posts to process.")
        return Counter()

    mode = "playwright" if args.use_playwright else "httpx"
    logging.info("Starting collection: posts=%s mode=%s concurrency=%s min_concurrency=%s", total, mode, args.concurrency, args.min_concurrency)

    controller = AdaptiveConcurrencyController(args)
    queue: asyncio.Queue[PostTask | None] = asyncio.Queue()
    result_queue: asyncio.Queue[PostResult | None] = asyncio.Queue()
    status_counts: Counter = Counter()
    client = None if args.use_playwright else await create_async_client(args.timeout)
    playwright = PlaywrightLoader(args) if args.use_playwright else None
    writer = IncrementalCsvWriter(args.output)
    start = time.monotonic()

    for task in tasks:
        await queue.put(task)
    for _ in range(args.concurrency):
        await queue.put(None)

    async def worker() -> None:
        while True:
            await controller.wait_if_paused()
            task = await queue.get()
            if task is None:
                queue.task_done()
                return
            try:
                result = await process_task(task, args, controller, client, playwright)
                await result_queue.put(result)
            except Exception as exc:
                now = datetime.now().isoformat(timespec="seconds")
                rows = [make_error_row(task.channel_username, task.post_id, task.tgstat_url, now, "request_error", f"Unexpected error: {type(exc).__name__}: {exc}")]
                await result_queue.put(PostResult(rows))
            finally:
                queue.task_done()

    async def csv_writer() -> None:
        pending: list[dict] = []
        pending_posts = 0
        done = 0
        try:
            while True:
                result = await result_queue.get()
                if result is None:
                    break
                done += result.post_count
                pending_posts += result.post_count
                pending.extend(result.rows)
                for row in result.rows:
                    status_counts[row.get("status", "")] += 1
                if result.rows and result.rows[0].get("status") != "ok":
                    row = result.rows[0]
                    logging.info("Fail: %s/%s status=%s error=%s", row["channel_username"], row["post_id"], row["status"], row["error"])
                if pending_posts >= args.batch_posts:
                    writer.write_rows(pending)
                    writer.file.flush()
                    pending.clear()
                    pending_posts = 0
                if done % 1000 == 0 or done == total:
                    elapsed = max(0.001, time.monotonic() - start)
                    rate = done / elapsed
                    eta = (total - done) / rate if rate else 0
                    logging.info("Progress: %s/%s posts, %.2f posts/s, ETA %.1f min", done, total, rate, eta / 60)
                result_queue.task_done()
            if pending:
                writer.write_rows(pending)
                writer.file.flush()
        finally:
            writer.close()

    workers = [asyncio.create_task(worker()) for _ in range(args.concurrency)]
    writer_task = asyncio.create_task(csv_writer())
    try:
        await queue.join()
        await asyncio.gather(*workers)
        await result_queue.put(None)
        await writer_task
    except KeyboardInterrupt:
        logging.warning("Interrupted. Saving collected rows and shutting down.")
        for item in workers:
            item.cancel()
        await result_queue.put(None)
        await writer_task
    finally:
        if client is not None:
            await client.aclose()
        if playwright is not None:
            await playwright.close()

    logging.info("Saved result CSV: %s", args.output)
    return status_counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    args = parse_args()
    validate_args(args)
    counts = asyncio.run(run_collector(args))
    for status in ["ok", "not_found", "forbidden", "no_chart_data", "request_error", "parse_error"]:
        logging.info("Summary %s: %s", status, counts.get(status, 0))


if __name__ == "__main__":
    main()
