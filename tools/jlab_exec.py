#!/usr/bin/env python3
"""Run commands inside an OpenI CloudBrain (JupyterLab) container from a shell.

CloudBrain debug tasks can only be created through the web UI, but the
JupyterLab server behind them exposes the standard REST + websocket API, so
everything after "the task is running" can be scripted: no screen, no clicking.

The base URL of a debug task looks like::

    https://<host>/notebook_<hash>_task0

Usage:
    export JLAB_BASE=https://101.69.200.162:30443/notebook_r5c32b333_task0
    python tools/jlab_exec.py --sh 'pip list | grep torch'
    python tools/jlab_exec.py 'print(1 + 1)'
    python tools/jlab_exec.py -f local_script.py
    python tools/jlab_exec.py --put local.py remote/sub/path.py
    python tools/jlab_exec.py --files
    python tools/jlab_exec.py --new-kernel --sh 'nvidia-smi'

Notes:
    * The server needs the ``_xsrf`` cookie for POST requests, otherwise the
      first kernel creation returns 403.
    * Ingress on the public endpoint occasionally resets TLS connections, so
      requests and websockets are retried.
    * A kernel executes one cell at a time; a long command blocks every later
      call to the same kernel.  Use ``--new-kernel`` when that happens.
"""
from __future__ import annotations

import argparse
import asyncio
import http.cookiejar
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

if not hasattr(urllib.request, "HTTPSHandler"):
    raise SystemExit("python is too old")

SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode = ssl.CERT_NONE

JAR = http.cookiejar.CookieJar()
OPENER = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(JAR),
    urllib.request.HTTPSHandler(context=SSL_CTX),
)

BASE = ""
SESSION = uuid.uuid4().hex
KERNEL_CACHE = Path(os.environ.get("JLAB_CACHE", Path.home() / ".cache" / "pymss" / "jlab_kernel"))


def api(path: str, payload: dict | None = None, method: str = "GET", retries: int = 6):
    """Call the Jupyter REST API, retrying through ingress resets."""
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    headers["Connection"] = "close"
    if method != "GET":
        bootstrap()
        headers["X-XSRFToken"] = _xsrf()
    request = urllib.request.Request(BASE + path, data=data, method=method, headers=headers)
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with OPENER.open(request, timeout=30) as response:
                body = response.read()
            return json.loads(body) if body else None
        except urllib.error.HTTPError as herr:
            if herr.code == 404:
                try:
                    body = herr.read()[:300]
                except Exception:
                    body = b"?"
                print(f"DBG404 {method} {path} url={request.full_url} body={body!r}", file=sys.stderr)
            raise
        except Exception as exc:  # noqa: BLE001 - linux ingress resets TLS often
            last = exc
            time.sleep(1.0 + 1.5 * attempt)
            if "xsrf" in headers:
                headers["X-XSRFToken"] = _xsrf()
    raise RuntimeError(f"{method} {path} failed: {type(last).__name__}: {last}")


def _xsrf() -> str:
    for cookie in JAR:
        if cookie.name == "_xsrf":
            return cookie.value
    return ""


def bootstrap() -> None:
    if _xsrf():
        return
    OPENER.open(BASE + "/lab", timeout=30).read()


def ensure_kernel(force_new: bool = False) -> str:
    if force_new:
        kernel_id = api("/api/kernels", {"name": "python3"}, "POST")["id"]
        _cache_kernel(kernel_id)
        return kernel_id
    if KERNEL_CACHE.exists():
        cached = KERNEL_CACHE.read_text().strip()
        if cached and any(kern["id"] == cached for kern in api("/api/kernels") or []):
            return cached
    for kern in api("/api/kernels") or []:
        if kern.get("execution_state") != "dead":
            _cache_kernel(kern["id"])
            return kern["id"]
    kernel_id = api("/api/kernels", {"name": "python3"}, "POST")["id"]
    _cache_kernel(kernel_id)
    return kernel_id


def _cache_kernel(kernel_id: str) -> None:
    try:
        KERNEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        KERNEL_CACHE.write_text(kernel_id)
    except OSError:
        pass


class Kernel:
    def __init__(self, kernel_id: str):
        self.kernel_id = kernel_id
        self.url = BASE.replace("https://", "wss://").replace("http://", "ws://") + (
            f"/api/kernels/{kernel_id}/channels?session_id={SESSION}"
        )

    async def __aenter__(self):
        import websockets

        headers = {"Origin": BASE}
        kwargs = (
            {"additional_headers": headers}
            if int(websockets.__version__.split(".")[0]) >= 14
            else {"extra_headers": headers}
        )
        last: Exception | None = None
        for attempt in range(4):
            try:
                self.ws = await websockets.connect(
                    self.url, ssl=SSL_CTX, max_size=None, open_timeout=30, ping_interval=None, **kwargs
                )
                return self
            except Exception as exc:  # noqa: BLE001
                last = exc
                await asyncio.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"websocket connect failed: {type(last).__name__}: {last}")

    async def __aexit__(self, *_exc):
        await self.ws.close()

    async def run(self, code: str, timeout: float = 3600.0) -> tuple[str, bool]:
        msg_id = uuid.uuid4().hex
        await self.ws.send(
            json.dumps(
                {
                    "header": {
                        "msg_id": msg_id,
                        "username": "jlab-exec",
                        "session": SESSION,
                        "msg_type": "execute_request",
                        "version": "5.3",
                    },
                    "parent_header": {},
                    "metadata": {},
                    "content": {
                        "code": code,
                        "silent": False,
                        "store_history": False,
                        "user_expressions": {},
                        "allow_stdin": False,
                        "stop_on_error": True,
                    },
                    "channel": "shell",
                }
            )
        )
        chunks: list[str] = []
        errored = False
        deadline = asyncio.get_event_loop().time() + timeout
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                chunks.append("\n[timeout]")
                return "".join(chunks), True
            reply = json.loads(await asyncio.wait_for(self.ws.recv(), timeout=remaining))
            if (reply.get("parent_header") or {}).get("msg_id") != msg_id:
                continue
            kind = reply.get("msg_type") or reply.get("header", {}).get("msg_type")
            content = reply.get("content") or {}
            if kind == "stream":
                chunks.append(content.get("text", ""))
            elif kind in ("execute_result", "display_data"):
                text = (content.get("data") or {}).get("text/plain")
                if text:
                    chunks.append(text if text.endswith("\n") else text + "\n")
            elif kind == "error":
                errored = True
                chunks.append("\n".join(content.get("traceback", [])))
            elif kind == "execute_reply":
                errored = errored or content.get("status") == "error"
                break
        return "".join(chunks), errored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("code", nargs="?", help="python code to execute in the container")
    parser.add_argument("--base", default=os.environ.get("JLAB_BASE", ""), help="notebook base URL (or JLAB_BASE)")
    parser.add_argument("--sh", metavar="CMD", help="run a shell command instead")
    parser.add_argument("-f", "--file", help="execute a local python file in the container")
    parser.add_argument("--put", nargs=2, metavar=("LOCAL", "REMOTE"), help="upload a file next to the server root")
    parser.add_argument("--files", action="store_true", help="list the server root directory")
    parser.add_argument("--new-kernel", action="store_true", help="start a fresh kernel instead of reusing one")
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()

    global BASE
    if not args.base:
        parser.error("set JLAB_BASE or pass --base (e.g. https://host/notebook_<hash>_task0)")
    BASE = args.base.rstrip("/")

    if args.files:
        for entry in api("/api/contents/")["content"]:
            print(f"{entry['type']:9s} {entry['name']}")
        return 0

    if args.put:
        local, remote = args.put
        content = Path(local).read_text(encoding="utf-8")
        api(f"/api/contents/{remote.lstrip('/')}", {"type": "file", "format": "text", "content": content}, "PUT")
        print(f"uploaded {local} -> {remote} ({len(content)} bytes)")
        return 0

    if args.sh:
        import base64

        encoded = base64.b64encode(args.sh.encode()).decode()
        code = (
            "import base64, subprocess\n"
            f"proc = subprocess.run(base64.b64decode('{encoded}').decode(), shell=True, capture_output=True, text=True)\n"
            "print(proc.stdout, end='')\n"
            "print(proc.stderr, end='', file=__import__('sys').stderr)\n"
            "print(f'[exit {proc.returncode}]')\n"
        )
    elif args.file:
        code = Path(args.file).read_text(encoding="utf-8")
    elif args.code:
        code = args.code
    else:
        parser.error("nothing to run: pass code, --sh, -f, --put or --files")
        return 2

    kernel_id = ensure_kernel(force_new=args.new_kernel)
    output, errored = asyncio.run(_run(kernel_id, code, args.timeout))
    print(output, end="")
    return 1 if errored else 0


async def _run(kernel_id: str, code: str, timeout: float) -> tuple[str, bool]:
    async with Kernel(kernel_id) as kernel:
        return await kernel.run(code, timeout=timeout)


if __name__ == "__main__":
    raise SystemExit(main())
