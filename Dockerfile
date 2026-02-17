# ---------------------------------------------------
# Base Image
# ---------------------------------------------------
FROM python:3.10-slim

# ---------------------------------------------------
# Environment settings
# ---------------------------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ---------------------------------------------------
# System dependencies (PIL/OpenCV + curl for healthcheck)
# ---------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------
# Create non-root user
# ---------------------------------------------------
RUN useradd -m -u 10001 appuser

# ---------------------------------------------------
# Set working directory
# ---------------------------------------------------
WORKDIR /app

# ---------------------------------------------------
# Install ONLY inference deps (avoid training bloat)
# ---------------------------------------------------
COPY requirements-api.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---------------------------------------------------
# Copy application code
# ---------------------------------------------------
COPY src ./src

# ---------------------------------------------------
# Bake model artifacts into image (required for reliable CD)
# ---------------------------------------------------
COPY models ./models

# ---------------------------------------------------
# Fix permissions for non-root runtime
# ---------------------------------------------------
RUN chown -R appuser:appuser /app

# ---------------------------------------------------
# Runtime configuration
# ---------------------------------------------------
EXPOSE 8000
USER appuser

# ---------------------------------------------------
# Healthcheck
# ---------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl -f http://localhost:8000/health || exit 1

# ---------------------------------------------------
# Start API
# ---------------------------------------------------
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
