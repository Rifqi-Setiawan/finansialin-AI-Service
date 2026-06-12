from fastapi import FastAPI, HTTPException, UploadFile, File
import uuid
import json
import os
import redis.asyncio as redis
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Finansialin OCR Service API")

# Initialize Redis client
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(redis_url, decode_responses=True)

@app.post("/ocr/jobs", status_code=202)
async def submit_ocr_job(receiptImage: UploadFile = File(...)):
    # Validasi
    if receiptImage.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Format gambar harus JPEG, PNG, atau WebP")
        
    contents = await receiptImage.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Ukuran gambar maksimal 5 MB")
        
    job_id = str(uuid.uuid4())
    
    # Simpan gambar sementara (di worker nanti akan dibaca)
    os.makedirs("/tmp/ocr_images", exist_ok=True)
    image_path = f"/tmp/ocr_images/{job_id}.img"
    with open(image_path, "wb") as f:
        f.write(contents)
        
    # Set status awal
    await redis_client.set(f"ocr:job:{job_id}", "queued", ex=3600)
    
    # Push ke queue
    job_data = {
        "job_id": job_id,
        "image_path": image_path
    }
    await redis_client.rpush("ocr:queue", json.dumps(job_data))
    
    return {"job_id": job_id, "status": "queued"}

@app.get("/ocr/jobs/{job_id}")
async def get_ocr_job_status(job_id: str):
    status = await redis_client.get(f"ocr:job:{job_id}")
    if not status:
        raise HTTPException(status_code=404, detail="Job tidak ditemukan")
        
    response = {"status": status}
    
    if status == "done":
        result = await redis_client.get(f"ocr:result:{job_id}")
        if result:
            response["result"] = json.loads(result)
            
    if status == "failed":
        error = await redis_client.get(f"ocr:error:{job_id}")
        if error:
            response["error"] = error
            
    return response

@app.get("/ocr/system")
async def get_system_status():
    status_str = await redis_client.get("ocr:system:status")
    if not status_str:
        return {"status": "unknown", "detail": "System status not available yet. Worker might be starting or down."}
        
    try:
        return json.loads(status_str)
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to parse system status.")
