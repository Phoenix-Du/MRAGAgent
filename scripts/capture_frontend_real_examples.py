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
LOG_DIR = ROOT / "logs"


def _health_ok(url: str) -> bool:
    try:
        resp = httpx.get(url, timeout=3)
        return 200 <= resp.status_code < 300
    except Exception:
        return False


def _start_service(name: str, health_url: str, args: list[str]) -> subprocess.Popen[str] | None:
    if _health_ok(health_url):
        return None
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"frontend_capture_{name}.log"
    log = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        args,
        cwd=str(ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    deadline = time.time() + 90
    while time.time() < deadline:
        if _health_ok(health_url):
            return proc
        if proc.poll() is not None:
            raise RuntimeError(f"{name} exited early; see {log_path}")
        time.sleep(1)
    raise TimeoutError(f"{name} did not become healthy; see {log_path}")


def _start_required_services(base_url: str) -> list[subprocess.Popen[str]]:
    procs: list[subprocess.Popen[str]] = []
    services = [
        (
            "raganything_bridge",
            "http://127.0.0.1:9002/healthz",
            [sys.executable, "-m", "uvicorn", "app.integrations.raganything_bridge:app", "--host", "127.0.0.1", "--port", "9002"],
        ),
        (
            "image_pipeline",
            "http://127.0.0.1:9010/healthz",
            [sys.executable, "-m", "uvicorn", "app.integrations.image_pipeline_bridge:app", "--host", "127.0.0.1", "--port", "9010"],
        ),
        (
            "orchestrator",
            f"{base_url}/healthz",
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"],
        ),
    ]
    for name, health, cmd in services:
        proc = _start_service(name, health, cmd)
        if proc is not None:
            procs.append(proc)
    return procs


def _prepare_capture_layout(page: Page, *, hide_images: bool = False) -> None:
    image_rule = ".images-row { display: none !important; }" if hide_images else ""
    page.add_style_tag(
        content=f"""
        html, body, .app, .main {{ height: auto !important; min-height: 100% !important; overflow: visible !important; }}
        .sidebar, .topbar, .composer-wrap, .thinking-wrap {{ display: none !important; }}
        .thread {{
          overflow: visible !important;
          height: auto !important;
          max-height: none !important;
          max-width: 1080px !important;
          padding: 28px 28px 36px !important;
        }}
        .msg {{ margin-bottom: 16px !important; }}
        .msg-bubble {{ max-width: 100% !important; }}
        .msg-role {{ display: none !important; }}
        .evidence-block {{ max-height: 260px; overflow: hidden; }}
        .images-row img {{ max-width: 260px !important; max-height: 190px !important; }}
        {image_rule}
        """
    )
    page.evaluate("() => window.scrollTo(0, 0)")


def _submit_and_capture(
    page: Page,
    *,
    base_url: str,
    intent: str,
    query: str,
    screenshot_path: Path,
    timeout_ms: int,
    extra_body: dict | None = None,
    hide_images: bool = False,
) -> None:
    page.goto(base_url, wait_until="domcontentloaded")
    if extra_body:
        page.route(
            "**/v1/chat/query",
            lambda route: route.continue_(
                post_data=__import__("json").dumps(
                    {
                        **__import__("json").loads(route.request.post_data or "{}"),
                        **extra_body,
                    },
                    ensure_ascii=False,
                )
            ),
        )
    page.locator("#intentSelect").select_option(intent)
    page.locator("#urlInput").fill("")
    page.locator("#queryInput").fill(query)

    before = page.locator(".msg-assistant").count()
    page.locator("#chatForm").evaluate("(form) => form.requestSubmit()")
    page.locator(".msg-assistant").nth(before).wait_for(state="visible", timeout=timeout_ms)
    page.wait_for_timeout(2000)
    _prepare_capture_layout(page, hide_images=hide_images)
    page.wait_for_timeout(800)
    screenshot_path.parent.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(screenshot_path), full_page=True)
    if extra_body:
        page.unroute("**/v1/chat/query")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    procs = _start_required_services(base_url)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)

            general_path = ASSET_DIR / "frontend_real_general_web_qa.png"
            image_path = ASSET_DIR / "frontend_real_image_search.png"

            _submit_and_capture(
                page,
                base_url=base_url,
                intent="general_qa",
                query="请自动检索网页并回答：FastAPI 是什么？它适合用来构建什么类型的应用？请用三点概括。",
                screenshot_path=general_path,
                timeout_ms=360_000,
                hide_images=True,
            )

            _submit_and_capture(
                page,
                base_url=base_url,
                intent="image_search",
                query="请帮我找一张红色汽车的图片，并说明返回图片为什么符合这个需求。",
                screenshot_path=image_path,
                timeout_ms=360_000,
            )

            browser.close()

        print(f"general_qa_screenshot={general_path}")
        print(f"image_search_screenshot={image_path}")
        return 0
    finally:
        for proc in reversed(procs):
            proc.terminate()
        for proc in reversed(procs):
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
