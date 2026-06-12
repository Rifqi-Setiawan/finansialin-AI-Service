import io
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from dotenv import load_dotenv
from PIL import Image

# Import schemas
from schemas import ChatRequest

# Import custom services
from services.chatbot import process_chat
from services.ocr import extract_receipt_data

# Import transformers with security bypass for Donut
import transformers.utils.import_utils
if hasattr(transformers.utils.import_utils, 'check_torch_load_is_safe'):
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
from transformers import DonutProcessor, VisionEncoderDecoderModel

# Global variables for OCR
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = None
model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global processor, model
    print(f"Loading Donut OCR model on {device} (Worker Process)...")
    try:
        processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2", use_safetensors=True)
        model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2", use_safetensors=True)
        model.to(device)
        print("Model loaded successfully into the active worker!")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Failed to load model: {e}")
        
    yield
    
    print("Shutting down AI service, clearing model from VRAM...")
    model = None
    processor = None

# Initialize FastAPI app
app = FastAPI(title="Finansialin AI Service", lifespan=lifespan)

# Load environment variables
load_dotenv()

@app.post("/predict/ocr")
async def predict_ocr(receiptImage: UploadFile = File(...)):
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model OCR gagal dimuat. Cek log terminal.")

    try:
        contents = await receiptImage.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"File gambar tidak valid: {str(e)}")

    try:
        result = extract_receipt_data(image, processor, model, device)
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"OCR Processing Error: {str(e)}")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"-> Menerima pesan dari User {request.user_id} (Session: {request.session_id})")
    print(f"-> Pesan: {request.message}")
    
    try:
        reply = process_chat(request.user_id, request.session_id, request.message)
        print(f"<- Balasan AI: {reply}")
        
        return {
            "reply": reply,
            "type": "text"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Gagal memproses pesan AI: {str(e)}")