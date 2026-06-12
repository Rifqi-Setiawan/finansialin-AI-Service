from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

# Import schemas
from schemas import ChatRequest

# Import custom services
from services.chatbot import process_chat, stream_chat, init_http_client, close_http_client
from services.memory import init_redis, close_redis
from contextlib import asynccontextmanager
from fastapi.responses import StreamingResponse

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_http_client()
    init_redis()
    yield
    await close_http_client()
    await close_redis()

# Initialize FastAPI app
app = FastAPI(title="Finansialin Chatbot Service", lifespan=lifespan)

# Load environment variables
load_dotenv()

@app.get("/health")
async def health_check():
    return {"status": "ok", "service": "chatbot-service"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    print(f"-> Menerima pesan dari User {request.user_id} (Session: {request.session_id})")
    print(f"-> Pesan: {request.message}")
    
    try:
        reply = await process_chat(request.user_id, request.session_id, request.message)
        print(f"<- Balasan AI: {reply}")
        
        return {
            "reply": reply,
            "type": "text"
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Gagal memproses pesan AI: {str(e)}")

@app.post("/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    print(f"-> Menerima pesan (STREAM) dari User {request.user_id} (Session: {request.session_id})")
    
    try:
        return StreamingResponse(
            stream_chat(request.user_id, request.session_id, request.message),
            media_type="text/event-stream"
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Gagal memulai stream AI: {str(e)}")
