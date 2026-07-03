"""Start the local MNEMOS + Ollama helper services for Open WebUI.

The script does not configure Open WebUI for the user. It starts the MNEMOS
research intake UI and the MNEMOS-backed Ollama/Open WebUI proxy, then prints
the exact Open WebUI connection settings to enter.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MNEMOS_BASE_URL = "http://127.0.0.1:8700"
DEFAULT_INTAKE_URL = "http://127.0.0.1:8788"
DEFAULT_PROXY_URL = "http://127.0.0.1:8790"
DEFAULT_OPENWEBUI_URL = "http://127.0.0.1:8088"


def normalize_base_url(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return text
    if "://" not in text:
        text = "http://" + text
    return text.rstrip("/").replace("http://0.0.0.0", "http://127.0.0.1", 1)


def default_ollama_base_url() -> str:
    explicit = os.getenv("OLLAMA_BASE_URL")
    if explicit:
        return normalize_base_url(explicit)
    host = os.getenv("OLLAMA_HOST")
    if host:
        return normalize_base_url(host)
    return "http://127.0.0.1:11434"


def http_json(url: str, *, timeout_s: float = 5.0) -> tuple[bool, str]:
    try:
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read(400).decode("utf-8", "replace")
            return True, f"{response.status} {body[:180]}"
    except URLError as exc:
        return False, str(exc)
    except Exception as exc:  # pragma: no cover - defensive CLI reporting.
        return False, str(exc)


def start_python_tool(
    tool: str,
    *,
    port: int,
    env: dict[str, str],
    extra_args: list[str] | None = None,
) -> subprocess.Popen[str]:
    command = [
        sys.executable,
        str(ROOT / "tools" / tool),
        "--port",
        str(port),
    ]
    if extra_args:
        command.extend(extra_args)
    return subprocess.Popen(
        command,
        cwd=str(ROOT),
        env=env,
        text=True,
    )


def print_status(label: str, ok: bool, detail: str) -> None:
    marker = "OK" if ok else "WARN"
    print(f"[{marker}] {label}: {detail}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnemos-base-url", default=os.getenv("MNEMOS_BASE_URL", DEFAULT_MNEMOS_BASE_URL))
    parser.add_argument("--ollama-base-url", default=default_ollama_base_url())
    parser.add_argument("--intake-port", type=int, default=8788)
    parser.add_argument("--proxy-port", type=int, default=8790)
    parser.add_argument("--openwebui-url", default=DEFAULT_OPENWEBUI_URL)
    parser.add_argument("--no-intake-ui", action="store_true")
    parser.add_argument("--no-proxy", action="store_true")
    parser.add_argument("--open-browser", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mnemos_base_url = normalize_base_url(args.mnemos_base_url)
    ollama_base_url = normalize_base_url(args.ollama_base_url)
    intake_url = f"http://127.0.0.1:{args.intake_port}"
    proxy_url = f"http://127.0.0.1:{args.proxy_port}"

    env = os.environ.copy()
    env["MNEMOS_BASE_URL"] = mnemos_base_url
    env["OLLAMA_BASE_URL"] = ollama_base_url

    print("MNEMOS local Open WebUI helper")
    print(f"MNEMOS_BASE_URL={mnemos_base_url}")
    print(f"OLLAMA_BASE_URL={ollama_base_url}")
    print()

    mnemos_ok, mnemos_detail = http_json(f"{mnemos_base_url}/health")
    print_status("MNEMOS health", mnemos_ok, mnemos_detail)
    ollama_ok, ollama_detail = http_json(f"{ollama_base_url}/api/tags")
    print_status("Ollama models", ollama_ok, ollama_detail)

    processes: list[subprocess.Popen[Any]] = []
    try:
        if not args.no_intake_ui:
            intake_ok, _ = http_json(intake_url)
            if intake_ok:
                print_status("Research intake UI", True, f"already running at {intake_url}")
            else:
                processes.append(
                    start_python_tool(
                        "mnemos_research_ui.py",
                        port=args.intake_port,
                        env=env,
                    )
                )
                time.sleep(1.5)
                print_status("Research intake UI", True, f"started at {intake_url}")

        if not args.no_proxy:
            proxy_ok, _ = http_json(f"{proxy_url}/health")
            if proxy_ok:
                print_status("Open WebUI proxy", True, f"already running at {proxy_url}")
            else:
                processes.append(
                    start_python_tool(
                        "mnemos_ollama_openwebui_proxy.py",
                        port=args.proxy_port,
                        env=env,
                    )
                )
                time.sleep(1.5)
                proxy_ok, proxy_detail = http_json(f"{proxy_url}/health")
                print_status("Open WebUI proxy", proxy_ok, proxy_detail)

        print()
        print("Open WebUI manual connection settings:")
        print("  Connection: OpenAI-compatible")
        print(f"  Base URL:   http://host.docker.internal:{args.proxy_port}/v1")
        print("  API Key:    mnemos-local")
        print("  Prefix ID:  mnemos")
        print()
        print("Use this page for artifact intake:")
        print(f"  {intake_url}")
        print("Use this page for chat after configuring Open WebUI:")
        print(f"  {args.openwebui_url}")

        if args.open_browser:
            webbrowser.open(intake_url)
            webbrowser.open(args.openwebui_url)

        if not processes:
            return 0

        print()
        print("Services started. Press Ctrl+C to stop services launched by this script.")
        while True:
            for process in list(processes):
                if process.poll() is not None:
                    print(f"Process exited: pid={process.pid} code={process.returncode}")
                    processes.remove(process)
            if not processes:
                return 0
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping services launched by this script...")
        for process in processes:
            if process.poll() is None:
                process.terminate()
        time.sleep(1)
        for process in processes:
            if process.poll() is None:
                process.kill()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
