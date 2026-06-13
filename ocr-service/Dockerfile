FROM python:3.11-slim

WORKDIR /app

# API image only needs FastAPI + Redis. The CUDA OCR worker runs natively on Windows.
ARG INSTALL_WORKER_DEPS=0

COPY requirements-api.txt requirements-api.txt
COPY requirements.txt requirements.txt

RUN if [ "$INSTALL_WORKER_DEPS" = "1" ]; then \
        apt-get update \
        && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
        && rm -rf /var/lib/apt/lists/* \
        && pip install --no-cache-dir -r requirements.txt; \
    else \
        pip install --no-cache-dir -r requirements-api.txt; \
    fi

# Copy source code
COPY . .

# Start script will be provided by docker-compose
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
