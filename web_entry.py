r"""Dedicated web entry point for Dawu Agent.

Starts the FastAPI backend, serves frontend static assets, and exposes
report downloads under a single process.

Usage:
    .venv\Scripts\python.exe web_entry.py
    # or with custom port:
    .venv\Scripts\python.exe web_entry.py --port 8080
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from dawu_agent.config.loader import ConfigLoader
from dawu_agent.server.app import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Dawu Agent Web UI")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()

    settings = ConfigLoader("config").load()
    app = create_app(settings)

    web_dir = Path(__file__).parent / "web"
    if not web_dir.exists():
        raise RuntimeError(f"Frontend directory not found: {web_dir}")

    # Explicit page routes must be registered before the catch-all static mount.
    @app.get("/")
    async def root() -> FileResponse:
        return FileResponse(web_dir / "login.html")

    @app.get("/login")
    async def login_page() -> FileResponse:
        return FileResponse(web_dir / "login.html")

    @app.get("/chat")
    async def chat_page() -> FileResponse:
        return FileResponse(web_dir / "chat.html")

    app.mount("/", StaticFiles(directory=web_dir), name="static")

    print(f"Starting Dawu Agent Web UI on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
