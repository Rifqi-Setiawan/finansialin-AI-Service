import os
import requests
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import create_agent
from typing import Optional

load_dotenv()

# 1. Dictionary untuk menyimpan riwayat chat sementara (Memory)
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = []
    return store[session_id]

# 2. Inisialisasi Model Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash", 
    temperature=0.3, 
    max_tokens=1024
)

# 3. Mendefinisikan Tools (Alat untuk AI)
LARAVEL_API_URL = os.getenv("LARAVEL_API_URL", "http://127.0.0.1:8000/api")

@tool
def get_recent_transactions(user_id: int, limit: int = 5) -> str:
    """
    Panggil alat ini untuk mendapatkan riwayat transaksi terakhir pengguna (pengeluaran dan pemasukan).
    Gunakan ini secara PROAKTIF saat pengguna meminta tips keuangan, saran penghematan, atau analisis pengeluaran, agar kamu memiliki data untuk memberikan saran yang relevan.
    """
    print(f"[TOOL DIPANGGIL] Mengambil {limit} transaksi terakhir untuk user_id: {user_id}")
    try:
        response = requests.get(
            f"{LARAVEL_API_URL}/internal/recent-transactions", 
            params={"user_id": user_id, "limit": limit}
        )
        if response.status_code == 200:
            return str(response.json())
        return "Sistem gagal mengambil riwayat transaksi."
    except Exception as e:
        return f"Error sistem internal: {str(e)}"

@tool
def get_user_balance(user_id: int) -> str:
    """
    Panggil alat ini untuk mengecek total saldo, jumlah uang, sisa uang di dompet, atau rekening pengguna.
    Gunakan alat ini secara PROAKTIF saat kamu butuh mengetahui kondisi keuangan pengguna untuk memberikan tips atau rekomendasi.
    """
    print(f"[TOOL DIPANGGIL] Mengambil data saldo untuk user_id: {user_id}")
    try:
        response = requests.get(f"{LARAVEL_API_URL}/internal/balance", params={"user_id": user_id})
        if response.status_code == 200:
            return str(response.json())
        return "Sistem gagal mengambil data saldo."
    except Exception as e:
        return f"Error sistem internal: {str(e)}"

@tool
def get_budget_status(user_id: int, month: Optional[int] = None, year: Optional[int] = None) -> str:
    """
    Panggil alat ini untuk mengetahui status budget (anggaran) pengguna, sisa budget, dan peringatan overbudget.
    Gunakan secara proaktif untuk memberi tips penghematan yang sesuai dengan batasan pengeluaran bulanan mereka.
    """
    print(f"[TOOL DIPANGGIL] Mengambil status budget untuk user_id: {user_id}, bulan: {month}, tahun: {year}")
    try:
        params = {"user_id": user_id}
        if month: params["month"] = month
        if year: params["year"] = year
        
        response = requests.get(f"{LARAVEL_API_URL}/internal/budget-status", params=params)
        if response.status_code == 200:
            return str(response.json())
        return "Sistem gagal mengambil data status budget."
    except Exception as e:
        return f"Error sistem internal: {str(e)}"

@tool
def get_monthly_analytics(user_id: int, month: Optional[int] = None, year: Optional[int] = None) -> str:
    """
    Panggil alat ini untuk mendapatkan ringkasan pengeluaran bulanan, pemasukan, dan pengeluaran terbesar.
    Gunakan secara proaktif saat memberikan evaluasi bulanan, mencari tahu pemborosan terbesar, atau menyusun strategi hemat.
    """
    print(f"[TOOL DIPANGGIL] Mengambil analitik bulanan untuk user_id: {user_id}, bulan: {month}, tahun: {year}")
    try:
        params = {"user_id": user_id}
        if month: params["month"] = month
        if year: params["year"] = year
        
        response = requests.get(f"{LARAVEL_API_URL}/internal/monthly-analytics", params=params)
        if response.status_code == 200:
            return str(response.json())
        return "Sistem gagal mengambil data analitik bulanan."
    except Exception as e:
        return f"Error sistem internal: {str(e)}"

@tool
def get_spending_trend(user_id: int, months: int = 3) -> str:
    """
    Gunakan alat ini saat pengguna bertanya tentang tren pengeluaran mereka selama beberapa bulan terakhir, 
    apakah mereka makin boros atau hemat, atau perbandingan pengeluaran antar bulan.
    """
    print(f"[TOOL DIPANGGIL] Mengambil tren pengeluaran untuk user_id: {user_id}, {months} bulan terakhir")
    try:
        response = requests.get(f"{LARAVEL_API_URL}/internal/spending-trend", params={"user_id": user_id, "months": months})
        if response.status_code == 200:
            return str(response.json())
        return "Sistem gagal mengambil data tren pengeluaran."
    except Exception as e:
        return f"Error sistem internal: {str(e)}"

@tool
def get_user_financial_profile(user_id: int) -> str:
    """
    Gunakan alat ini untuk mengambil ringkasan profil keuangan pengguna secara keseluruhan 
    (seperti total aset, kekayaan bersih/net worth, atau rasio tabungan).
    """
    print(f"[TOOL DIPANGGIL] Mengambil profil finansial untuk user_id: {user_id}")
    try:
        response = requests.get(f"{LARAVEL_API_URL}/internal/financial-profile", params={"user_id": user_id})
        if response.status_code == 200:
            return str(response.json())
        return "Sistem gagal mengambil data profil finansial."
    except Exception as e:
        return f"Error sistem internal: {str(e)}"

@tool
def get_savings_goals(user_id: int) -> str:
    """
    Gunakan alat ini saat pengguna bertanya tentang target tabungan mereka, progress menabung, 
    goals, atau seberapa dekat mereka mencapai tujuan keuangan (contoh: tabungan liburan, beli laptop).
    """
    print(f"[TOOL DIPANGGIL] Mengambil target tabungan untuk user_id: {user_id}")
    try:
        response = requests.get(f"{LARAVEL_API_URL}/internal/savings-goals", params={"user_id": user_id})
        if response.status_code == 200:
            return str(response.json())
        return "Sistem gagal mengambil data target tabungan."
    except Exception as e:
        return f"Error sistem internal: {str(e)}"

tools = [
    get_user_balance, 
    get_recent_transactions, 
    get_budget_status, 
    get_monthly_analytics, 
    get_spending_trend, 
    get_user_financial_profile, 
    get_savings_goals
]

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
agent = create_agent(llm, tools, system_prompt=system_instruction)

def process_chat(user_id: int, session_id: str, message: str) -> str:
    """
    Memproses pesan masuk dari pengguna dan mengembalikan balasan dari agen AI.
    """
    history = get_session_history(session_id)
    
    # INJEKSI KONTEKS
    contextual_message = f"[Sistem: Ingat, user_id pengguna yang sedang ngobrol denganmu saat ini adalah {user_id}].\n\n{message}"
    history.append(HumanMessage(content=contextual_message))
    
    # Eksekusi AI Agent 
    response = agent.invoke({"messages": history})
    
    # Ekstrak balasan AI
    raw_content = response["messages"][-1].content
    if isinstance(raw_content, list):
        ai_reply = " ".join([item.get("text", "") for item in raw_content if isinstance(item, dict) and "text" in item])
    else:
        ai_reply = str(raw_content)
        
    # Simpan balasan AI ke history
    history.append(AIMessage(content=ai_reply))
    
    return ai_reply
