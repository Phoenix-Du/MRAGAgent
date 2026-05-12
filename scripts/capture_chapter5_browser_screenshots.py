from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import httpx
from playwright.sync_api import Page, sync_playwright


ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "docs" / "architecture" / "assets" / "chapter5"
BASE_URL = "http://127.0.0.1:8000"


def _health_ok(base_url: str) -> bool:
    try:
      r = httpx.get(f"{base_url}/healthz", timeout=3)
      return r.status_code == 200
    except Exception:
      return False


def _start_server(base_url: str) -> subprocess.Popen[str] | None:
    if _health_ok(base_url):
        return None

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / "chapter5_browser_capture_uvicorn.log"
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=str(ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.time() + 60
    while time.time() < deadline:
        if _health_ok(base_url):
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"uvicorn exited early; see {log_path}")
        time.sleep(1)
    raise TimeoutError(f"server did not become healthy within 60s; see {log_path}")


def _submit_query(
    page: Page,
    *,
    base_url: str,
    intent: str,
    query: str,
    screenshot_path: Path,
    url: str | None = None,
    timeout_ms: int = 180_000,
) -> None:
    page.goto(base_url, wait_until="domcontentloaded")
    page.locator("#intentSelect").select_option(intent)
    page.locator("#urlInput").fill(url or "")
    page.locator("#queryInput").fill(query)

    before = page.locator(".msg-assistant").count()
    page.locator("#chatForm").evaluate("(form) => form.requestSubmit()")
    page.locator(".msg-assistant").nth(before).wait_for(state="visible", timeout=timeout_ms)
    page.wait_for_timeout(1200)

    thread = page.locator("#thread")
    thread.evaluate("(el) => { el.scrollTop = el.scrollHeight; }")
    page.wait_for_timeout(300)
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(screenshot_path), full_page=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture real browser screenshots for thesis chapter 5.")
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    server_proc = _start_server(base_url)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)

            general_path = ASSET_DIR / "chapter5_browser_general_qa.png"
            image_path = ASSET_DIR / "chapter5_browser_image_search.png"

            _submit_query(
                page,
                base_url=base_url,
                intent="general_qa",
                query="请概括这个网页的主要内容，并说明它适合作为什么测试页面。",
                url="https://example.com/",
                screenshot_path=general_path,
            )

            page.locator("#btnNewChat").click()
            page.wait_for_timeout(300)

            _submit_query(
                page,
                base_url=base_url,
                intent="image_search",
                query="请找一张红色圆形图标或红色圆形物体的图片，并说明图片是否满足要求。",
                screenshot_path=image_path,
            )

            browser.close()

        print(f"general_qa_screenshot={general_path}")
        print(f"image_search_screenshot={image_path}")
        return 0
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
