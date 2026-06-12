# PROMPT UNTUK AI AGENT: Refactor Tahap 1 Finansialin AI Service

Salin seluruh isi di bawah garis ini ke AI coding agent kamu.

---

## PERAN DAN KONTEKS

Kamu adalah senior backend engineer yang bertugas merefactor AI service untuk aplikasi manajemen keuangan bernama **Finansialin**. Sistem terdiri dari dua repository:

1. **Backend Laravel** (repo: `finansialin-backend-laravel`): Laravel 12, PostgreSQL, REST API untuk auth, transaksi, budget, notifikasi. Menyediakan endpoint internal `/api/internal/*` (recent-transactions, balance, budget-status, monthly-analytics, spending-trend, financial-profile, savings-goals) yang dipanggil oleh AI service.
2. **AI Service** (repo: `finansialin-AI-Service`): FastAPI Python. Berisi dua fitur: (a) OCR struk belanja menggunakan model Donut `naver-clova-ix/donut-base-finetuned-cord-v2` via HuggingFace Transformers + torch, di-load saat startup di `main.py`; (b) chatbot finansial menggunakan LangChain + Gemini (`gemini-2.5-flash`) dengan 7 tools di `services/chatbot.py` yang masing-masing memanggil endpoint internal Laravel via library `requests` (blocking). Chat history disimpan di dictionary Python in-memory (`store = {}`) dengan key `session_id`.

## MASALAH YANG HARUS DISELESAIKAN

1. Model Donut OCR berat, di-load dalam proses FastAPI yang sama dengan chatbot, dan inferensi berjalan synchronous sehingga memblokir worker.
2. Endpoint `/chat` adalah `async def` tetapi memanggil `process_chat` yang sync dan menggunakan `requests` blocking, sehingga event loop terblokir dan concurrency hancur.
3. Chat history in-memory: hilang saat restart, tidak konsisten jika dijalankan multi-worker.
4. Agent dapat memanggil banyak tool berurutan, setiap tool adalah satu round-trip HTTP ke Laravel, menambah latensi.
5. Tidak ada streaming response, user menunggu jawaban penuh.
6. Endpoint `/api/internal/*` di Laravel tidak memiliki autentikasi service-to-service.

## TARGET ARSITEKTUR

Pecah AI service menjadi dua service terpisah dalam satu repo (struktur monorepo sederhana):

```
finansialin-AI-Service/
├── chatbot-service/        # FastAPI, ringan, I/O-bound, tanpa torch
│   ├── main.py
│   ├── schemas.py
│   ├── services/
│   │   ├── chatbot.py
│   │   └── memory.py       # Redis chat history
│   ├── requirements.txt
│   └── Dockerfile
├── ocr-service/            # FastAPI + worker, job-based
│   ├── main.py             # endpoint submit job + cek status
│   ├── worker.py           # konsumer job OCR
│   ├── services/ocr.py
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml      # chatbot, ocr-api, ocr-worker, redis
└── README.md
```

Redis digunakan untuk dua hal: chat history (chatbot-service) dan job queue + penyimpanan hasil OCR (ocr-service).

## TUGAS DETAIL

Kerjakan berurutan. Setiap tugas harus selesai dan bisa dijalankan sebelum lanjut ke tugas berikutnya.

### Tugas 1: Pisahkan chatbot-service dari OCR

1. Buat folder `chatbot-service/` berisi FastAPI app baru yang HANYA memuat endpoint `/chat` dan `/health`. Tidak boleh ada import torch, transformers, atau PIL di service ini.
2. `requirements.txt` chatbot-service hanya berisi: fastapi, uvicorn[standard], pydantic, python-dotenv, httpx, redis, langchain, langchain-core, langchain-google-genai.
3. Pertahankan kontrak request/response yang ada agar Laravel tidak perlu berubah: request `{ user_id, session_id, message }`, response `{ reply, type }`. Tambahkan varian streaming di Tugas 4.

### Tugas 2: Jadikan chatbot benar-benar async

1. Ganti semua pemanggilan `requests.get(...)` di dalam tools menjadi `httpx.AsyncClient` dengan timeout eksplisit (connect 3 detik, read 10 detik) dan satu client instance yang dibuat saat startup (lifespan) lalu di-reuse, bukan dibuat per request.
2. Ubah seluruh tool LangChain menjadi async function (gunakan `@tool` pada `async def`).
3. Ubah `process_chat` menjadi `async def` dan gunakan `ainvoke` / `astream` pada agent, bukan `invoke`.
4. Tambahkan retry sederhana (maksimal 2 retry dengan backoff) untuk panggilan ke Laravel, dan pastikan kegagalan satu tool mengembalikan pesan error yang informatif ke agent tanpa membuat seluruh request gagal.

### Tugas 3: Pindahkan chat history ke Redis

1. Buat modul `services/memory.py` dengan fungsi `get_history(session_id)` dan `append_history(session_id, role, content)`.
2. Simpan history sebagai Redis list dengan key `chat:history:{session_id}`, setiap item adalah JSON `{ "role": "...", "content": "...", "ts": ... }`.
3. Set TTL 30 menit pada key, di-refresh setiap kali ada pesan baru.
4. Batasi history yang dikirim ke LLM maksimal 10 pesan terakhir (LTRIM saat append).
5. Gunakan library `redis` dengan client async (`redis.asyncio`). URL Redis dari env var `REDIS_URL`.
6. Hapus total dictionary `store = {}` dan semua pemakaiannya.

### Tugas 4: Streaming response untuk chatbot

1. Tambahkan endpoint baru `POST /chat/stream` yang mengembalikan Server-Sent Events (SSE) via `StreamingResponse` dengan media type `text/event-stream`.
2. Stream token dari Gemini menggunakan `astream` / `astream_events` LangChain. Format event: `data: {"delta": "..."}` per chunk, diakhiri `data: {"done": true, "reply": "<jawaban penuh>"}`.
3. Endpoint `/chat` lama tetap ada untuk kompatibilitas (non-streaming).
4. Setelah stream selesai, simpan pesan user dan jawaban penuh AI ke Redis history.

### Tugas 5: Kurangi round-trip tool dengan context injection

1. Di repo Laravel, buat endpoint baru `GET /api/internal/financial-context?user_id=X` yang mengembalikan dalam satu response: saldo, 5 transaksi terakhir, status budget bulan berjalan, dan ringkasan analytics bulan berjalan. Implementasinya cukup memanggil ulang logic/service yang sudah dipakai endpoint internal yang ada, jangan duplikasi query mentah di controller.
2. Di chatbot-service, saat request chat masuk, panggil endpoint ini SATU KALI di awal (sebelum agent jalan), lalu suntikkan hasilnya ke system prompt sebagai konteks keuangan user.
3. Ringkas daftar tools dari 7 menjadi 3 saja, untuk data yang tidak tercakup konteks awal: `get_spending_trend`, `get_savings_goals`, dan `get_recent_transactions` (untuk kasus user minta lebih dari 5 transaksi). Tool lain dihapus karena datanya sudah ada di konteks.
4. Jika panggilan financial-context gagal, chatbot tetap berjalan tanpa konteks (graceful degradation), jangan error 500.

### Tugas 6: Refactor OCR menjadi job-based async service

1. Buat folder `ocr-service/` dengan dua proses:
   - `main.py` (API): endpoint `POST /ocr/jobs` menerima upload gambar, validasi (maks 5 MB, format JPEG/PNG/WebP), simpan gambar sementara, push job ke Redis queue, balas 202 dengan `{ job_id, status: "queued" }`. Endpoint `GET /ocr/jobs/{job_id}` mengembalikan `{ status: queued|processing|done|failed, result?, error? }`. API ini TIDAK meng-import torch.
   - `worker.py`: loop yang mengonsumsi job dari Redis queue (gunakan BLPOP), load model Donut SEKALI saat worker start, proses gambar, simpan hasil ke Redis key `ocr:result:{job_id}` dengan TTL 1 jam.
2. Gunakan Redis murni untuk queue (list `ocr:queue`) agar dependensi minimal. Jangan tambahkan Celery kecuali diminta.
3. Status job disimpan di Redis key `ocr:job:{job_id}` dan diupdate worker di setiap fase.
4. Pertahankan logic ekstraksi di `services/ocr.py` apa adanya, hanya pindahkan pemanggilan modelnya ke worker.
5. Tambahkan env var `OCR_ENGINE` dengan nilai `donut` (default) atau `gemini`. Jika `gemini`, worker tidak load Donut sama sekali, melainkan mengirim gambar ke Gemini Flash multimodal dengan instruksi mengembalikan JSON terstruktur `{ merchant, date, items: [{name, qty, price}], total }` menggunakan structured output. Ini menyiapkan jalur migrasi keluar dari torch.

### Tugas 7: Amankan komunikasi service-to-service

1. Di Laravel, buat middleware `InternalServiceAuth` yang memeriksa header `X-Internal-Token` terhadap env `INTERNAL_API_TOKEN`. Pasang pada seluruh route group `/api/internal/*`. Balas 401 jika tidak cocok.
2. Di chatbot-service, kirim header tersebut pada setiap panggilan httpx ke Laravel, nilai dari env `INTERNAL_API_TOKEN`.
3. Tambahkan kedua env var ke `.env.example` di masing-masing repo.

### Tugas 8: Docker compose dan dokumentasi

1. Buat `docker-compose.yml` di root repo AI service dengan services: `redis`, `chatbot` (port 8001), `ocr-api` (port 8002), `ocr-worker` (tanpa port, replicas bisa dinaikkan). Semua membaca env dari file `.env`.
2. Dockerfile chatbot harus image ringan (python slim, tanpa torch). Dockerfile OCR boleh berat karena berisi torch.
3. Update README dengan: diagram arsitektur sederhana (teks/ASCII), cara menjalankan via docker compose, daftar env var, contoh curl untuk `/chat`, `/chat/stream`, dan alur job OCR (submit lalu polling).

## ENVIRONMENT VARIABLES YANG HARUS DIDUKUNG

```
GOOGLE_API_KEY=
LARAVEL_API_URL=http://host.docker.internal:8000/api
REDIS_URL=redis://redis:6379/0
INTERNAL_API_TOKEN=
OCR_ENGINE=donut
```

## ATURAN DAN BATASAN

1. Jangan mengubah kontrak response endpoint `/chat` yang lama. Frontend dan Laravel yang sudah ada harus tetap berfungsi tanpa perubahan, kecuali penambahan header internal token.
2. Jangan menambahkan dependensi besar yang tidak disebut (tanpa Celery, tanpa Kafka, tanpa database baru).
3. Semua konfigurasi lewat environment variable, tidak ada nilai hardcoded (URL, token, model name).
4. Ganti semua `print()` dengan logging modul `logging` Python berformat terstruktur, sertakan `session_id`/`job_id` di setiap log.
5. Setiap endpoint harus punya error handling eksplisit, jangan biarkan traceback mentah bocor ke response.
6. Tulis docstring singkat di setiap fungsi publik. Komentar kode dalam Bahasa Indonesia.
7. Jangan menulis ulang logic prompt/persona chatbot yang sudah ada di `services/chatbot.py`, pertahankan perilakunya, hanya ubah infrastruktur di sekitarnya.

## DEFINITION OF DONE

1. `docker compose up` menjalankan redis, chatbot, ocr-api, dan ocr-worker tanpa error.
2. `POST /chat` merespons normal dan history bertahan setelah service di-restart (selama TTL belum habis).
3. `POST /chat/stream` mengalirkan token secara bertahap (bisa diuji dengan `curl -N`).
4. `POST /ocr/jobs` langsung balas 202 dalam waktu di bawah 500 ms, dan `GET /ocr/jobs/{id}` akhirnya mengembalikan status `done` beserta hasil ekstraksi.
5. Memanggil `/api/internal/balance` tanpa header `X-Internal-Token` menghasilkan 401, dengan header valid menghasilkan 200.
6. Tidak ada pemanggilan `requests` (library sync) tersisa di chatbot-service, verifikasi dengan grep.
7. Endpoint `/chat` tetap responsif saat ada job OCR sedang berjalan (uji dengan menjalankan keduanya bersamaan).

Kerjakan tugas satu per satu. Setelah menyelesaikan setiap tugas, jalankan dan verifikasi sendiri sebelum lanjut, lalu laporkan ringkasan perubahan file per tugas.
