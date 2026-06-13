import os
import json
import time
import threading
import traceback
from io import BytesIO
import redis
from PIL import Image, ImageOps
from dotenv import load_dotenv

import torch
from check_gpu import check_gpu

from transformers import DonutProcessor, VisionEncoderDecoderModel
from transformers.utils import import_utils

load_dotenv()

check_gpu()

# Redis sync client for worker
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.from_url(redis_url, decode_responses=True)

# Optimization settings
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

processor = None
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

def resolve_image_path(image_path: str) -> str:
    api_image_dir = os.getenv("OCR_IMAGE_DIR", "/tmp/ocr_images").replace("\\", "/").rstrip("/")
    worker_image_dir = os.getenv("OCR_WORKER_IMAGE_DIR", "").strip()
    normalized_path = image_path.replace("\\", "/")

    if worker_image_dir and normalized_path.startswith(f"{api_image_dir}/"):
        relative_path = normalized_path[len(api_image_dir):].lstrip("/")
        return os.path.join(worker_image_dir, *relative_path.split("/"))

    return image_path

def init_donut_model():
    global processor, model
    if hasattr(import_utils, 'check_torch_load_is_safe'):
        import_utils.check_torch_load_is_safe = lambda: None
    
    print(f"Loading Donut OCR model on {device} in fp16...")
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2", use_safetensors=True)
    model = VisionEncoderDecoderModel.from_pretrained(
        "naver-clova-ix/donut-base-finetuned-cord-v2", 
        use_safetensors=True,
        torch_dtype=dtype
    )
    model.to(device)
    model.eval()
    print("Donut model loaded successfully!")
    
    # Warm-up
    print("Running warmup inference...")
    dummy_image = Image.new('RGB', (800, 800), color='white')
    with torch.inference_mode():
        pixel_values = processor(dummy_image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device, dtype=dtype)
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(device)
        model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=50,
            num_beams=1,
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    print("Warmup complete!")

init_donut_model()
from services.ocr import extract_receipt_data

def preprocess_image(image: Image.Image) -> Image.Image:
    # Auto-rotate based on EXIF
    image = ImageOps.exif_transpose(image)
    image = image.convert("RGB")
    
    # Resize keeping aspect ratio, max side 1280
    max_size = 1280
    if max(image.size) > max_size:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return image

# System monitor thread
def system_monitor():
    while True:
        try:
            vram_used = 0
            vram_total = 0
            if device == "cuda":
                free, total = torch.cuda.mem_get_info()
                vram_used = total - free
                vram_total = total
                
            queue_len = redis_client.llen("ocr:queue")
            
            # get avg ms
            recent_ms = redis_client.lrange("ocr:metrics:inference_ms", 0, -1)
            avg_ms = 0
            if recent_ms:
                avg_ms = sum(float(x) for x in recent_ms) / len(recent_ms)
                
            status_data = {
                "device": device,
                "vram_used_gb": round(vram_used / (1024**3), 2),
                "vram_total_gb": round(vram_total / (1024**3), 2),
                "queue_size": queue_len,
                "avg_inference_ms": round(avg_ms, 2)
            }
            redis_client.set("ocr:system:status", json.dumps(status_data), ex=60)
        except Exception as e:
            pass
        time.sleep(30)

threading.Thread(target=system_monitor, daemon=True).start()

print("Worker started. Waiting for jobs...")

while True:
    try:
        job = redis_client.blpop("ocr:queue", timeout=0)
        if not job:
            continue
            
        _, job_data_str = job
        job_data = json.loads(job_data_str)
        job_id = job_data["job_id"]
        image_path = resolve_image_path(job_data["image_path"])
        
        print(f"\nProcessing job: {job_id}")
        redis_client.set(f"ocr:job:{job_id}", "processing", ex=3600)
        
        t0 = time.time()
        preprocessing_ms = 0
        inference_ms = 0
        parsing_ms = 0
        
        try:
            # Preprocessing
            t_pre = time.time()
            raw_image = Image.open(image_path)
            processed_image = preprocess_image(raw_image)
            preprocessing_ms = (time.time() - t_pre) * 1000
            
            # Extraction logic runs inside extract_receipt_data, which we will modify
            result, timings = extract_receipt_data(processed_image, processor, model, device)
            
            inference_ms = timings.get('inference_ms', 0)
            parsing_ms = timings.get('parsing_ms', 0)
            
            total_ms = (time.time() - t0) * 1000
            
            print(f"Timing - Pre: {preprocessing_ms:.1f}ms | Inf: {inference_ms:.1f}ms | Parse: {parsing_ms:.1f}ms | Total: {total_ms:.1f}ms")
            
            # Record metrics
            redis_client.lpush("ocr:metrics:inference_ms", inference_ms)
            redis_client.ltrim("ocr:metrics:inference_ms", 0, 9)
            
            redis_client.set(f"ocr:result:{job_id}", json.dumps(result), ex=3600)
            redis_client.set(f"ocr:job:{job_id}", "done", ex=3600)
            
        except torch.cuda.OutOfMemoryError as e:
            redis_client.set(f"ocr:error:{job_id}", "Out of VRAM", ex=3600)
            redis_client.set(f"ocr:job:{job_id}", "failed", ex=3600)
            print("ERROR: Out of VRAM! Recovering...")
            if device == "cuda":
                torch.cuda.empty_cache()
                
        except Exception as e:
            traceback.print_exc()
            redis_client.set(f"ocr:error:{job_id}", str(e), ex=3600)
            redis_client.set(f"ocr:job:{job_id}", "failed", ex=3600)
            print(f"Job {job_id} failed: {e}")
            
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)
                
            # VRAM management
            if device == "cuda":
                free, _ = torch.cuda.mem_get_info()
                if free < 1024 * 1024 * 1024:  # Less than 1GB free
                    print("Low VRAM detected. Emptying cache...")
                    torch.cuda.empty_cache()
                    
    except Exception as e:
        print(f"Worker loop error: {e}")
        time.sleep(1)
