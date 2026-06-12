# Finansialin AI Service

This repository contains the AI microservices for the Finansialin application. It is split into two main services: `chatbot-service` and `ocr-service`.

## Architecture

```text
finansialin-AI-Service
├── chatbot-service (FastAPI, Port 8001)
│   └── Handles LLM Chat & Tools (Gemini)
├── ocr-service (FastAPI API + Background Worker, Port 8002)
│   └── Handles Receipt OCR extraction (Donut GPU Accelerated)
└── redis (Port 6379)
    └── Message Broker (Job Queue) & Chat History Storage
```

## Running the Services

### Prerequisites & CUDA Optimization (For OCR Worker)
The OCR Worker relies heavily on GPU acceleration for fast inference. By default, it expects a CUDA-compatible NVIDIA GPU (like RTX 3050 Laptop).
1. Ensure your machine has NVIDIA drivers installed.
2. When running natively on Windows, install PyTorch with CUDA support explicitly inside your `ocr-service` environment:
   ```bash
   cd ocr-service
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
3. **Docker vs Host Setup**:
   - For `redis`, `chatbot-service`, and `ocr-api`, using Docker Compose is perfectly fine.
   - For `ocr-worker`, running it directly on your Windows host (`python worker.py`) is recommended for the best VRAM access and simplest setup without configuring WSL2 GPU Passthrough.

### Running with Docker Compose

You can run the entire stack (or partial stack) using Docker Compose:

```bash
# Clone the repository and configure env
cp .env.example .env
# Edit .env with your keys

# Build and start services
docker compose up -d --build
```

*(Note: If you run `ocr-worker` locally on Windows, you can remove it from docker-compose.yml or scale it to 0 `docker compose up --scale ocr-worker=0`)*

### Environment Variables

Required environment variables in `.env`:
- `GOOGLE_API_KEY`: API Key for Gemini.
- `LARAVEL_API_URL`: URL to the Laravel Backend.
- `REDIS_URL`: Redis Connection URL.
- `INTERNAL_API_TOKEN`: Secret token for service-to-service auth with Laravel.
- `OCR_MAX_CONCURRENCY`: Explicitly set to `1` (worker limits processing to 1 concurrent job to save VRAM).

## API Usage Examples

### 1. Chatbot - Normal Request

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "session_id": "session-123",
    "message": "Berapa saldo saya?"
  }'
```

### 2. Chatbot - Streaming Request (SSE)

```bash
curl -N -X POST http://localhost:8001/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "session_id": "session-123",
    "message": "Berapa saldo saya?"
  }'
```

### 3. OCR - Submit Job

```bash
curl -X POST http://localhost:8002/ocr/jobs \
  -H "Content-Type: multipart/form-data" \
  -F "receiptImage=@/path/to/struk.jpg"
```
Response:
```json
{
  "job_id": "1234-abcd...",
  "status": "queued"
}
```

### 4. OCR - Poll Job Status

```bash
curl -X GET http://localhost:8002/ocr/jobs/1234-abcd...
```

### 5. OCR - System Metrics & VRAM

```bash
curl -X GET http://localhost:8002/ocr/system
```
Response:
```json
{
  "device": "cuda",
  "vram_used_gb": 1.25,
  "vram_total_gb": 4.0,
  "queue_size": 0,
  "avg_inference_ms": 850.5
}
```
