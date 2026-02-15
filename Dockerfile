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
# System dependencies (needed for PIL / OpenCV / TF)
# ---------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------
# Create non-root user (security best practice)
# ---------------------------------------------------
RUN useradd -m appuser

# ---------------------------------------------------
# Set working directory
# ---------------------------------------------------
WORKDIR /app

# ---------------------------------------------------
# Install Python dependencies first (cache layer)
# ---------------------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---------------------------------------------------
# Copy application source
# ---------------------------------------------------
COPY src ./src

# ---------------------------------------------------
# (OPTIONAL) If you want model baked into image
# Uncomment next line ONLY if models exist during build
# ---------------------------------------------------
# COPY models ./models

# ---------------------------------------------------
# Runtime configuration
# ---------------------------------------------------
EXPOSE 8000

USER appuser

# ---------------------------------------------------
# Healthcheck (used by Docker/K8s)
# ---------------------------------------------------
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
  CMD curl -f http://localhost:8000/health || exit 1

# ---------------------------------------------------
# Start API
# ---------------------------------------------------
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
