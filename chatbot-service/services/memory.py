import os
import json
import time
import redis.asyncio as redis
from langchain_core.messages import HumanMessage, AIMessage

# Global Redis client
redis_client = None

def init_redis():
    global redis_client
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    redis_client = redis.from_url(redis_url, decode_responses=True)

async def close_redis():
    global redis_client
    if redis_client:
        await redis_client.aclose()
        redis_client = None

async def get_history(session_id: str):
    """Mengambil history chat dari Redis dan mengembalikan list Langchain Messages."""
    if not redis_client:
        return []
    
    key = f"chat:history:{session_id}"
    raw_history = await redis_client.lrange(key, 0, -1)
    
    messages = []
    for item in raw_history:
        try:
            data = json.loads(item)
            if data.get("role") == "user":
                messages.append(HumanMessage(content=data.get("content")))
            elif data.get("role") == "ai":
                messages.append(AIMessage(content=data.get("content")))
        except Exception as e:
            print(f"Error parsing history item: {e}")
            
    return messages

async def append_history(session_id: str, role: str, content: str):
    """Menyimpan pesan baru ke Redis history dengan TTL 30 menit dan batas 10 pesan."""
    if not redis_client:
        return
        
    key = f"chat:history:{session_id}"
    message = json.dumps({
        "role": role,
        "content": content,
        "ts": int(time.time())
    })
    
    # RPUSH ke list
    await redis_client.rpush(key, message)
    # Batasi 10 pesan terakhir
    await redis_client.ltrim(key, -10, -1)
    # Set TTL 30 menit (1800 detik)
    await redis_client.expire(key, 1800)
