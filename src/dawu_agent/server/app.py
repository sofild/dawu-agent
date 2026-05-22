"""FastAPI application factory with health checks and API routes."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dawu_agent.config.loader import Settings
from dawu_agent.observability.telemetry import TelemetryManager


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


def create_app(settings: Settings) -> FastAPI:
    """Create and configure the FastAPI application."""
    telemetry = TelemetryManager(settings)

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
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        return generate_latest()

    return app
