"""Enterprise observability with OpenTelemetry and Prometheus."""

from __future__ import annotations

import atexit
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import structlog

from dawu_agent.config.loader import Settings


class TelemetryManager:
    """Manages OpenTelemetry tracers, meters, and structured logging."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tracer: trace.Tracer | None = None
        self._meter: metrics.Meter | None = None
        self._logger: Any = None
        self._initialized = False

    def initialize(self) -> None:
        """Initialize all observability components."""
        if self._initialized:
            return

        resource = Resource.create({
            SERVICE_NAME: getattr(self.settings.logging, 'opentelemetry_service_name', 'dawu-agent'),
            SERVICE_VERSION: "0.1.0",
        })

        # Traces - only enable if endpoint is reachable (not localhost in dev)
        otlp_endpoint = getattr(self.settings.logging, 'opentelemetry_endpoint', '')
        enable_otlp = (self.settings.logging.opentelemetry_enabled
                       and otlp_endpoint
                       and not otlp_endpoint.startswith('http://localhost'))

        if enable_otlp:
            try:
                trace_provider = TracerProvider(resource=resource)
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, timeout=2)
                trace_provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                trace.set_tracer_provider(trace_provider)
                self._tracer = trace.get_tracer(__name__)
            except Exception:
                self._tracer = trace.get_tracer(__name__)
        else:
            # Use no-op tracer for local dev
            trace_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(trace_provider)
            self._tracer = trace.get_tracer(__name__)

        # Metrics - disabled for local dev to avoid export errors
        self._meter = metrics.get_meter(__name__)

        # Structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer() if self.settings.logging.format == "json"
                else structlog.dev.ConsoleRenderer(),
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        self._logger = structlog.get_logger()

        self._initialized = True
        atexit.register(self.shutdown)

    @property
    def tracer(self) -> trace.Tracer:
        if self._tracer is None:
            raise RuntimeError("Telemetry not initialized")
        return self._tracer

    @property
    def meter(self) -> metrics.Meter:
        if self._meter is None:
            raise RuntimeError("Telemetry not initialized")
        return self._meter

    @property
    def logger(self) -> Any:
        if self._logger is None:
            raise RuntimeError("Telemetry not initialized")
        return self._logger

    def shutdown(self) -> None:
        """Gracefully shutdown telemetry providers."""
        if self._tracer:
            trace_provider = trace.get_tracer_provider()
            if hasattr(trace_provider, "shutdown"):
                trace_provider.shutdown()
        if self._meter:
            metrics_provider = metrics.get_meter_provider()
            if hasattr(metrics_provider, "shutdown"):
                metrics_provider.shutdown()
