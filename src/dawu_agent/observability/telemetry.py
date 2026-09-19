"""Enterprise observability with OpenTelemetry and Prometheus."""

from __future__ import annotations

import atexit
from typing import Any

import structlog
from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from dawu_agent.config.loader import Settings


class TelemetryManager:
    """Manages OpenTelemetry tracers, meters, and structured logging."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._tracer: trace.Tracer | None = None
        self._meter: metrics.Meter | None = None
        self._logger: Any = None
        self._initialized = False
        self._quality_metrics: QualityMetrics | None = None

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

    @property
    def quality_metrics(self) -> QualityMetrics:
        if self._quality_metrics is None:
            self._quality_metrics = QualityMetrics()
        return self._quality_metrics

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


class QualityMetrics:
    """Eight quality indicators for agent observability (v4)."""

    def __init__(self) -> None:
        self._task_completion_rate: list[bool] = []
        self._tool_selection_accuracy: list[bool] = []
        self._param_error_count: int = 0
        self._trajectory_consistency: list[bool] = []
        self._cost_per_task: list[float] = []
        self._latency_samples: list[float] = []
        self._error_recovery_count: int = 0
        self._error_recovery_success: int = 0

    def record_task_completion(self, success: bool) -> None:
        self._task_completion_rate.append(success)

    def record_tool_selection(self, correct: bool) -> None:
        self._tool_selection_accuracy.append(correct)

    def record_param_error(self) -> None:
        self._param_error_count += 1

    def record_trajectory_consistency(self, consistent: bool) -> None:
        self._trajectory_consistency.append(consistent)

    def record_cost(self, cost: float) -> None:
        self._cost_per_task.append(cost)

    def record_latency(self, seconds: float) -> None:
        self._latency_samples.append(seconds)

    def record_error_recovery(self, success: bool) -> None:
        self._error_recovery_count += 1
        if success:
            self._error_recovery_success += 1

    def snapshot(self) -> dict[str, Any]:
        """Return current metrics as a dict for logging/Prometheus."""
        total_tasks = len(self._task_completion_rate)
        return {
            "task_completion_rate": (
                sum(self._task_completion_rate) / total_tasks
                if total_tasks else 0.0
            ),
            "tool_selection_accuracy": (
                sum(self._tool_selection_accuracy) / len(self._tool_selection_accuracy)
                if self._tool_selection_accuracy else 0.0
            ),
            "param_error_count": self._param_error_count,
            "trajectory_consistency_rate": (
                sum(self._trajectory_consistency) / len(self._trajectory_consistency)
                if self._trajectory_consistency else 0.0
            ),
            "avg_cost_per_task": (
                sum(self._cost_per_task) / len(self._cost_per_task)
                if self._cost_per_task else 0.0
            ),
            "latency_p50": self._percentile(self._latency_samples, 50),
            "latency_p99": self._percentile(self._latency_samples, 99),
            "error_recovery_rate": (
                self._error_recovery_success / self._error_recovery_count
                if self._error_recovery_count else 0.0
            ),
            "total_tasks": total_tasks,
        }

    @staticmethod
    def _percentile(data: list[float], pct: int) -> float:
        if not data:
            return 0.0
        sorted_data = sorted(data)
        n = len(sorted_data)
        index = int(pct / 100 * n)
        if index >= n:
            index = n - 1
        return sorted_data[index]
