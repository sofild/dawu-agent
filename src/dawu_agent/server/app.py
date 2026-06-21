"""FastAPI application factory with health checks, API routes, and web UI backend."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from dawu_agent.config.loader import Settings
from dawu_agent.core.agent import Agent
from dawu_agent.core.events import (
    AgentEvent,
    AssistantTextEvent,
    CompactionEvent,
    ErrorEvent,
    FinalResponseEvent,
    StateChangeEvent,
    ToolResultEvent,
    ToolUseEvent,
    TurnEndEvent,
    TurnStartEvent,
    UserMessageEvent,
)
from dawu_agent.observability.telemetry import TelemetryManager


# ---------------------------------------------------------------------------
# Request/response schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tokens_used: int = 0


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"


class LoginRequest(BaseModel):
    password: str


class LoginResponse(BaseModel):
    success: bool
    message: str


# ---------------------------------------------------------------------------
# Simple signed-session manager (no extra dependencies)
# ---------------------------------------------------------------------------
class SessionManager:
    """HMAC-SHA256 signed session cookie manager."""

    COOKIE_NAME = "dawu_session"

    def __init__(self, settings: Settings) -> None:
        password = os.getenv("WEB_UI_PASSWORD", "")
        explicit_secret = os.getenv("WEB_SESSION_SECRET", "")
        if explicit_secret:
            self._secret = explicit_secret.encode()
        elif password:
            self._secret = hashlib.sha256(password.encode()).digest()
        else:
            # Fallback for development only; sessions do not survive restarts.
            self._secret = os.urandom(32)

    def create_session(self, data: dict[str, Any]) -> str:
        """Sign session data and return a cookie-safe string."""
        payload = base64.urlsafe_b64encode(
            json.dumps(data, ensure_ascii=False).encode()
        ).decode()
        sig = hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()
        return f"{payload}.{sig}"

    def verify_session(self, cookie_value: str | None) -> dict[str, Any] | None:
        """Verify signature and return session payload."""
        if not cookie_value or "." not in cookie_value:
            return None
        payload, sig = cookie_value.rsplit(".", 1)
        expected = hmac.new(self._secret, payload.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        try:
            return json.loads(base64.urlsafe_b64decode(payload.encode()).decode())
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Event serialization helpers
# ---------------------------------------------------------------------------
_FILE_PATH_RE = re.compile(
    r"(?:saved to|saved as|written to|report saved to|chart saved to|file written successfully)\s*[:：]\s*(.+?)(?:\n|$)",
    re.IGNORECASE,
)


def _event_to_dict(event: AgentEvent) -> dict[str, Any]:
    """Convert an AgentEvent dataclass to a JSON-serializable dict."""
    result: dict[str, Any] = {"type": event.type, "timestamp": event.timestamp}
    for attr in (
        "old",
        "new",
        "reason",
        "turn",
        "duration",
        "content",
        "is_final",
        "tool_name",
        "tool_input",
        "tool_use_id",
        "is_error",
        "error_type",
        "action",
        "detail",
        "text",
    ):
        if hasattr(event, attr):
            value = getattr(event, attr)
            if value is not None:
                result[attr] = value
    return result


def _extract_file_paths(text: str, project_root: Path) -> list[str]:
    """Extract file paths from tool result text and make them relative."""
    paths: list[str] = []
    for match in _FILE_PATH_RE.finditer(text):
        path = match.group(1).strip().strip("'\"`")
        if not path:
            continue
        # Normalize Windows backslashes.
        path = path.replace("\\", "/")
        # If an absolute/Windows path is given, try to make it project-relative.
        p = Path(path)
        if p.is_absolute():
            try:
                rel = p.resolve().relative_to(project_root.resolve())
                path = _normalize_path(str(rel))
            except ValueError:
                # Path is outside project root; keep as-is and let security check filter it.
                pass
        paths.append(path)
    return paths


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


# ---------------------------------------------------------------------------
# File scanner for report/download discovery
# ---------------------------------------------------------------------------
_DOWNLOAD_ROOT_NAMES = ("reports", "workspace", "sessions")


def _get_download_roots(project_root: Path) -> list[Path]:
    roots: list[Path] = []
    for name in _DOWNLOAD_ROOT_NAMES:
        p = project_root / name
        if p.exists():
            roots.append(p.resolve())
    return roots


def _is_under_download_roots(target: Path, roots: list[Path]) -> bool:
    resolved = target.resolve()
    for root in roots:
        try:
            resolved.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _scan_new_files(
    roots: list[Path], session_start: float, seen: set[str], project_root: Path
) -> list[dict[str, str]]:
    """Scan allowed directories for files newer than session_start."""
    new_files: list[dict[str, str]] = []
    for root in roots:
        if not root.exists():
            continue
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            try:
                mtime = f.stat().st_mtime
            except OSError:
                continue
            key = str(f.resolve())
            if mtime > session_start and key not in seen:
                seen.add(key)
                try:
                    rel = f.resolve().relative_to(project_root.resolve())
                    path_str = _normalize_path(str(rel))
                except ValueError:
                    path_str = _normalize_path(str(f))
                new_files.append(
                    {
                        "path": path_str,
                        "name": f.name,
                    }
                )
    return new_files


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(settings: Settings) -> FastAPI:
    """Create and configure the FastAPI application."""
    telemetry = TelemetryManager(settings)
    session_manager = SessionManager(settings)
    agent_sessions: dict[str, Agent] = {}
    project_root = Path(".").resolve()
    download_roots = _get_download_roots(project_root)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        telemetry.initialize()
        telemetry.logger.info("server.starting")
        yield
        telemetry.logger.info("server.shutting_down")
        telemetry.shutdown()

    app = FastAPI(
        title="Dawu Agent API",
        description="Enterprise AI Agent for data analysis and report generation",
        version="0.1.0",
        lifespan=lifespan,
    )

    # -----------------------------------------------------------------------
    # Auth helpers
    # -----------------------------------------------------------------------
    async def require_auth(request: Request) -> dict[str, Any]:
        session = session_manager.verify_session(request.cookies.get(SessionManager.COOKIE_NAME))
        if not session:
            raise HTTPException(status_code=401, detail="Not authenticated")
        return session

    @app.post("/api/auth/login", response_model=LoginResponse)
    async def login(
        request: LoginRequest,
        response: Response,
    ) -> LoginResponse:
        expected = os.getenv("WEB_UI_PASSWORD", "")
        if not expected:
            raise HTTPException(status_code=500, detail="WEB_UI_PASSWORD not configured")
        if request.password != expected:
            raise HTTPException(status_code=401, detail="Invalid password")
        token = session_manager.create_session({"auth": True, "ts": time.time()})
        response.set_cookie(
            key=SessionManager.COOKIE_NAME,
            value=token,
            httponly=True,
            max_age=7 * 24 * 3600,
            samesite="lax",
        )
        return LoginResponse(success=True, message="Login successful")

    @app.post("/api/auth/logout")
    async def logout(response: Response) -> dict[str, str]:
        response.delete_cookie(SessionManager.COOKIE_NAME)
        return {"message": "Logged out"}

    @app.get("/api/auth/me")
    async def me(user: dict[str, Any] = Depends(require_auth)) -> dict[str, Any]:
        return {"authenticated": True}

    # -----------------------------------------------------------------------
    # Existing endpoints
    # -----------------------------------------------------------------------
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(status="healthy")

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        # TODO: Integrate with Agent core
        return ChatResponse(
            response=f"Echo: {request.message}",
            session_id=request.session_id or "new-session",
        )

    @app.get("/metrics")
    async def metrics() -> str:
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

        return generate_latest()

    # -----------------------------------------------------------------------
    # Chat streaming (SSE)
    # -----------------------------------------------------------------------
    async def _ensure_agent(web_session_id: str) -> Agent:
        agent = agent_sessions.get(web_session_id)
        if agent is None:
            agent = Agent(settings=settings, telemetry=telemetry)
            await agent.initialize()
            agent_sessions[web_session_id] = agent
        return agent

    async def _chat_stream_generator(
        web_session_id: str, message: str
    ) -> AsyncGenerator[str, None]:
        agent = await _ensure_agent(web_session_id)
        session_start = time.time()
        seen_files: set[str] = set()
        loop = asyncio.get_running_loop()

        # Initial scan to ignore pre-existing files.
        _scan_new_files(download_roots, session_start, seen_files, project_root)

        try:
            async for event in agent.run_stream(message, continue_session=True):
                event_dict = _event_to_dict(event)
                yield f"data: {json.dumps(event_dict, ensure_ascii=False)}\n\n"

                # Detect files announced in tool results.
                if isinstance(event, ToolResultEvent):
                    for path in _extract_file_paths(event.content or "", project_root):
                        full = (project_root / path).resolve()
                        if _is_under_download_roots(full, download_roots):
                            key = str(full)
                            if key not in seen_files:
                                seen_files.add(key)
                                yield f"data: {json.dumps({'type': 'file_ready', 'path': path, 'name': Path(path).name}, ensure_ascii=False)}\n\n"

                # Opportunistic directory scan between events.
                new_files = await loop.run_in_executor(
                    None, _scan_new_files, download_roots, session_start, seen_files, project_root
                )
                for f in new_files:
                    yield f"data: {json.dumps({'type': 'file_ready', **f}, ensure_ascii=False)}\n\n"

            # Final directory scan after the stream ends.
            new_files = await loop.run_in_executor(
                None, _scan_new_files, download_roots, session_start, seen_files, project_root
            )
            for f in new_files:
                yield f"data: {json.dumps({'type': 'file_ready', **f}, ensure_ascii=False)}\n\n"

        except Exception as e:
            telemetry.logger.error("server.stream_error", error=str(e))
            yield f"data: {json.dumps({'type': 'error', 'error_type': 'stream_error', 'detail': str(e)}, ensure_ascii=False)}\n\n"

    @app.get("/api/chat/stream")
    async def chat_stream(
        request: Request,
        message: str,
        session_id: str | None = None,
        user: dict[str, Any] = Depends(require_auth),
    ) -> StreamingResponse:
        if not message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")

        web_session_id = session_id or user.get("sid")
        if not web_session_id:
            # Generate a deterministic ID for this browser session if missing.
            web_session_id = request.cookies.get(SessionManager.COOKIE_NAME, "")[:16]

        return StreamingResponse(
            _chat_stream_generator(web_session_id, message),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post("/api/chat/clear")
    async def chat_clear(
        request: Request,
        session_id: str | None = None,
        user: dict[str, Any] = Depends(require_auth),
    ) -> dict[str, str]:
        cookie_fallback = request.cookies.get(SessionManager.COOKIE_NAME, "")[:16]
        web_session_id = session_id or user.get("sid") or cookie_fallback
        agent = agent_sessions.pop(web_session_id, None)
        if agent:
            await agent.shutdown()
        return {"message": "Session cleared"}

    # -----------------------------------------------------------------------
    # File download
    # -----------------------------------------------------------------------
    @app.get("/api/download")
    async def download(
        path: str,
        user: dict[str, Any] = Depends(require_auth),
    ) -> FileResponse:
        target = project_root / path
        if not _is_under_download_roots(target, download_roots):
            raise HTTPException(status_code=403, detail="Access denied")
        if not target.exists() or not target.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(
            target,
            filename=target.name,
            content_disposition_type="attachment",
        )

    return app
