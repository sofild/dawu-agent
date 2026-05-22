# =============================================================================
# Dawu Agent - Production Docker Image
# =============================================================================

FROM python:3.12-slim as builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir hatchling && \
    pip install --no-cache-dir -e "."

# =============================================================================
# Production stage
# =============================================================================
FROM python:3.12-slim

WORKDIR /app

# Create non-root user
RUN groupadd -r dawu && useradd -r -g dawu dawu

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create runtime directories
RUN mkdir -p /app/workspace /app/sessions /app/memory /app/logs && \
    chown -R dawu:dawu /app

USER dawu

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

ENTRYPOINT ["dawu"]
CMD ["server", "--host", "0.0.0.0", "--port", "8000"]
