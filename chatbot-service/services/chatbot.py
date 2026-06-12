import os
import asyncio
import httpx
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from typing import Optional

load_dotenv()

from .memory import get_history, append_history

# 2. Inisialisasi Model Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.3, 
    max_tokens=1024
)

LARAVEL_API_URL = os.getenv("LARAVEL_API_URL", "http://127.0.0.1:8000/api")

# HTTPX AsyncClient Global
http_client: httpx.AsyncClient = None

def init_http_client():
    global http_client
    timeout = httpx.Timeout(10.0, connect=3.0)
    http_client = httpx.AsyncClient(timeout=timeout)

async def close_http_client():
    global http_client
    if http_client:
        await http_client.aclose()
        http_client = None

async def fetch_laravel_api(endpoint: str, params: dict) -> str:
    """Helper function to fetch from Laravel with retry and backoff."""
    max_retries = 2
    token = os.getenv("INTERNAL_API_TOKEN", "")
    headers = {"X-Internal-Token": token}
    
    for attempt in range(max_retries + 1):
        try:
            response = await http_client.get(f"{LARAVEL_API_URL}{endpoint}", params=params, headers=headers)
            if response.status_code == 200:
                return str(response.json())
            return f"Sistem gagal mengambil data (Status Code: {response.status_code})."
        except Exception as e:
            if attempt < max_retries:
                await asyncio.sleep(2 ** attempt)  # Backoff
            else:
                return f"Error sistem internal setelah retry: {str(e)}"

# 3. Mendefinisikan Tools (Alat untuk AI)
@tool
async def get_recent_transactions(user_id: int, limit: int = 5) -> str:
    """
    Panggil alat ini untuk mendapatkan riwayat transaksi terakhir pengguna (pengeluaran dan pemasukan).
    Gunakan ini secara PROAKTIF saat pengguna meminta tips keuangan, saran penghematan, atau analisis pengeluaran.
    """
    print(f"[TOOL DIPANGGIL] Mengambil {limit} transaksi terakhir untuk user_id: {user_id}")
    return await fetch_laravel_api("/internal/recent-transactions", {"user_id": user_id, "limit": limit})

@tool
async def get_spending_trend(user_id: int, months: int = 3) -> str:
    """
    Gunakan alat ini saat pengguna bertanya tentang tren pengeluaran mereka selama beberapa bulan terakhir.
    """
    print(f"[TOOL DIPANGGIL] Mengambil tren pengeluaran untuk user_id: {user_id}, {months} bulan terakhir")
    return await fetch_laravel_api("/internal/spending-trend", {"user_id": user_id, "months": months})

@tool
async def get_savings_goals(user_id: int) -> str:
    """
    Gunakan alat ini saat pengguna bertanya tentang target tabungan mereka, progress menabung, goals.
    """
    print(f"[TOOL DIPANGGIL] Mengambil target tabungan untuk user_id: {user_id}")
    return await fetch_laravel_api("/internal/savings-goals", {"user_id": user_id})

tools = [
    get_recent_transactions, 
    get_spending_trend, 
    get_savings_goals
]

async def get_financial_context(user_id: int) -> str:
    """Mengambil konteks keuangan pengguna untuk diinjeksi ke prompt."""
    result = await fetch_laravel_api("/internal/financial-context", {"user_id": user_id})
    if "Sistem gagal" in result or "Error sistem" in result:
        return "Data konteks keuangan saat ini tidak tersedia."
    return result

# 4. System Prompt Baru
system_instruction = """Kamu adalah **Finansialin AI** — asisten keuangan pribadi yang cerdas, empatik, dan proaktif.

Aturan Penting:
1. WAJIB MENGGUNAKAN TOOLS SECARA PROAKTIF: Jika kamu membutuhkan konteks keuangan pengguna (seperti saldo, transaksi, atau status budget) untuk memberikan saran yang kontekstual (seperti saat diminta tips hemat, analisis, atau pertanyaan umum), KAMU WAJIB memanggil tools yang tersedia pada giliran ini juga! JANGAN meminta pengguna untuk menunggu atau mengatakan kamu akan memanggilnya nanti. Langsung panggil tools tersebut. Jangan pernah menebak data keuangan.
2. Gunakan bahasa Indonesia yang kasual, hangat, dan bersahabat (gunakan 'aku' dan 'kamu').
3. Setelah mendapat data dari tool, sampaikan datanya dengan rapi dan ramah (tambahkan format Rupiah yang benar).
4. Jika menampilkan riwayat transaksi, gunakan format bullet point (bullet points) agar mudah dibaca, sebutkan nama kategori/merchant dan nominalnya.
5. Jika pengguna bertanya soal budget, periksa statusnya dan berikan peringatan dengan nada suportif jika mereka mendekati atau sudah melewati batas (overbudget).
6. Saat memberikan rekomendasi penghematan atau strategi mencapai target, berikan langkah-langkah yang SPESIFIK dan ACTIONABLE. Contoh: Sebutkan kategori mana yang harus dipangkas berdasarkan data pengeluaran terbesarnya, berikan batas nominal angka yang realistis, dan gunakan prinsip keuangan dasar jika diperlukan.
"""

# 5. Buat Agent
agent = create_react_agent(llm, tools, state_modifier=system_instruction)

async def process_chat(user_id: int, session_id: str, message: str) -> str:
    """
    Memproses pesan masuk dari pengguna dan mengembalikan balasan dari agen AI secara asinkron.
    """
    history = await get_history(session_id)
    financial_context = await get_financial_context(user_id)
    
    # INJEKSI KONTEKS
    contextual_message = f"[Sistem: Ingat, user_id pengguna yang sedang ngobrol denganmu saat ini adalah {user_id}.\nKonteks Keuangan Saat Ini: {financial_context}]\n\n{message}"
    history.append(HumanMessage(content=contextual_message))
    
    # Eksekusi AI Agent secara async
    response = await agent.ainvoke({"messages": history})
    
    # Ekstrak balasan AI
    raw_content = response["messages"][-1].content
    if isinstance(raw_content, list):
        ai_reply = " ".join([item.get("text", "") for item in raw_content if isinstance(item, dict) and "text" in item])
    else:
        ai_reply = str(raw_content)
        
    # Simpan balasan AI ke history
    await append_history(session_id, "user", message)
    await append_history(session_id, "ai", ai_reply)
    
    return ai_reply

import json

async def stream_chat(user_id: int, session_id: str, message: str):
    """
    Memproses pesan masuk dan mengembalikan response stream menggunakan Server-Sent Events (SSE).
    """
    history = await get_history(session_id)
    financial_context = await get_financial_context(user_id)
    
    contextual_message = f"[Sistem: Ingat, user_id pengguna yang sedang ngobrol denganmu saat ini adalah {user_id}.\nKonteks Keuangan Saat Ini: {financial_context}]\n\n{message}"
    history.append(HumanMessage(content=contextual_message))
    
    full_reply = ""
    
    # Gunakan astream_events untuk mendapatkan token stream
    async for event in agent.astream_events(
        {"messages": history},
        version="v1"
    ):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if isinstance(chunk, list):
                # Sometimes gemini returns list of chunks
                text_chunk = " ".join([item.get("text", "") for item in chunk if isinstance(item, dict) and "text" in item])
            else:
                text_chunk = str(chunk)
                
            if text_chunk:
                full_reply += text_chunk
                yield f"data: {json.dumps({'delta': text_chunk})}\n\n"
                
    # Stream selesai
    yield f"data: {json.dumps({'done': True, 'reply': full_reply})}\n\n"
    
    # Simpan balasan AI ke history setelah stream selesai
    await append_history(session_id, "user", message)
    await append_history(session_id, "ai", full_reply)
