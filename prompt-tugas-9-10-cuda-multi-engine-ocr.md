# PROMPT ADDENDUM: Tugas 9-10 Optimasi CUDA dan Multi-Engine OCR

Lanjutan dari prompt Tahap 1 Finansialin AI Service. Salin seluruh isi di bawah garis ini ke AI coding agent, dikerjakan SETELAH Tugas 1-8 selesai.

---

## KONTEKS TAMBAHAN

Hardware development: laptop Windows dengan GPU NVIDIA RTX 3050 Laptop (4 GB VRAM), dipakai untuk menjalankan ocr-worker secara lokal. Model OCR saat ini Donut `naver-clova-ix/donut-base-finetuned-cord-v2` (sekitar 200M parameter, muat sangat lega di 4 GB VRAM dalam fp16). Target: inferensi OCR per struk di bawah 1 detik di GPU, tanpa out-of-memory, dan worker tetap stabil dipakai berjam-jam.

## TUGAS 9: Aktifkan dan optimalkan CUDA untuk Donut di ocr-worker

### 9.1 Instalasi dan verifikasi CUDA

1. Pastikan `requirements.txt` ocr-service TIDAK memaksa torch versi CPU. Tambahkan instruksi instalasi di README: torch harus diinstal dari index CUDA resmi PyTorch (pilih build cu121 atau yang lebih baru sesuai driver), contoh perintah pip dengan `--index-url https://download.pytorch.org/whl/cu121`.
2. Buat script `check_gpu.py` di ocr-service yang mencetak: `torch.cuda.is_available()`, nama GPU, total dan free VRAM, versi CUDA runtime. Worker harus menjalankan pengecekan ini saat startup dan log hasilnya. Jika CUDA tidak tersedia, log WARNING jelas lalu fallback ke CPU, jangan crash.

### 9.2 Optimasi loading dan inferensi model

Terapkan semua poin berikut di `worker.py` / `services/ocr.py`:

1. Load model sekali saat worker start dengan `torch_dtype=torch.float16` dan pindahkan ke `cuda`. Gunakan `model.eval()`.
2. Bungkus seluruh inferensi dengan `torch.inference_mode()` (bukan hanya `no_grad`).
3. Pastikan input pixel_values dikonversi ke `torch.float16` dan dipindah ke device yang sama sebelum `model.generate`.
4. Set parameter generate untuk kecepatan: `num_beams=1` (greedy), `use_cache=True`, `early_stopping` sesuai, dan `max_length` secukupnya untuk struk (jangan default panjang).
5. Aktifkan `torch.backends.cudnn.benchmark = True` dan `torch.backends.cuda.matmul.allow_tf32 = True` di awal worker.
6. Tambahkan WARM-UP: setelah model load, jalankan satu inferensi dummy (gambar putih kecil) supaya request pertama user tidak kena cold-start kernel CUDA.
7. Preprocessing gambar SEBELUM masuk processor: resize sisi terpanjang maksimal 1280 px dengan menjaga aspect ratio, konversi RGB, auto-rotate berdasarkan EXIF orientation. Foto kamera HP 4000 px tidak boleh masuk mentah.
8. Setelah setiap job selesai, panggil `torch.cuda.empty_cache()` HANYA jika free VRAM di bawah ambang 1 GB (cek dengan `torch.cuda.mem_get_info`), jangan setiap job karena justru memperlambat.
9. Batasi worker OCR ke 1 proses dengan 1 copy model (concurrency diatur oleh antrian Redis, bukan paralelisme model). Tambahkan env `OCR_MAX_CONCURRENCY=1` untuk dokumentasi eksplisit.
10. Tambahkan pengukuran waktu per fase di log setiap job: preprocessing_ms, inference_ms, parsing_ms, total_ms. Ini wajib supaya optimasi bisa diverifikasi dengan angka.

### 9.3 Stabilitas di laptop

1. Tangkap `torch.cuda.OutOfMemoryError` secara eksplisit: tandai job `failed` dengan pesan jelas, kosongkan cache, dan worker lanjut hidup memproses job berikutnya, jangan mati.
2. Tambahkan endpoint `GET /ocr/system` di ocr-api yang membaca status dari Redis (di-update worker tiap 30 detik): device aktif, VRAM used/total, jumlah job di antrian, rata-rata inference_ms 10 job terakhir.
3. Catatan untuk Docker: GPU passthrough di Windows membutuhkan Docker Desktop dengan backend WSL2 dan menambahkan `deploy.resources.reservations.devices` (driver nvidia) di service ocr-worker pada docker-compose. Sediakan juga jalur alternatif: jalankan ocr-worker langsung di host Windows (tanpa Docker) sementara redis, chatbot, dan ocr-api tetap di Docker. Dokumentasikan kedua cara di README dan jadikan jalur host-native sebagai default development karena paling sederhana.

## TUGAS 10: Multi-engine OCR (donut, paddleocr-vl, gemini)

1. Refactor `OCR_ENGINE` menjadi tiga nilai: `donut` (default), `paddleocr_vl`, `gemini`. Buat abstraksi sederhana: satu interface `BaseOcrEngine` dengan method `extract(image) -> dict` berskema seragam `{ merchant, date, items: [{name, qty, price}], total, raw_text?, confidence? }`. Tiga implementasi terpisah dalam folder `engines/`.
2. Engine `donut`: implementasi hasil Tugas 9, lalu mapping output token CORD ke skema seragam.
3. Engine `paddleocr_vl`: gunakan paket resmi PaddleOCR terbaru dengan dukungan GPU. Model hanya di-load saat engine ini dipilih (lazy import, jangan import paddle saat engine donut aktif). Hasil parsing teks struk kemudian dipetakan ke skema seragam dengan parsing rule sederhana (regex total, tanggal, baris item).
4. Engine `gemini`: kirim gambar ke Gemini Flash dengan structured output (response schema JSON sesuai skema seragam). Gunakan model dari env `GEMINI_OCR_MODEL`.
5. Hanya SATU engine yang di-load per proses worker sesuai env. Mengganti engine dilakukan dengan restart worker, bukan runtime switching, agar VRAM 4 GB tidak diperebutkan dua model.
6. Buat script benchmark `benchmark_ocr.py`: menerima folder berisi gambar struk uji, menjalankan engine yang aktif terhadap semua gambar, dan mengeluarkan CSV berisi nama file, total_ms, hasil ekstraksi JSON. Tujuannya membandingkan donut vs paddleocr_vl vs gemini dengan struk Indonesia asli secara objektif.

## ATURAN

1. Jangan menambahkan TensorRT, ONNX export, atau torch.compile dulu. Fp16 + CUDA + greedy decoding sudah cukup untuk target sub-detik; kompleksitas tambahan tidak sepadan di tahap ini.
2. Semua angka ambang (max image size, VRAM threshold, max_length) lewat env var dengan default yang disebut di atas.
3. Jangan ubah kontrak job API dari Tugas 6 (`POST /ocr/jobs`, `GET /ocr/jobs/{id}`).
4. requirements ocr-service dipecah: `requirements-base.txt` (fastapi, redis, pillow), `requirements-donut.txt` (torch, transformers, sentencepiece), `requirements-paddle.txt` (paddleocr, paddlepaddle-gpu). README menjelaskan kombinasi instalasi per engine.

## DEFINITION OF DONE

1. Log worker saat startup menampilkan GPU terdeteksi (nama RTX 3050) dan model load dalam fp16 di cuda.
2. Job OCR struk uji selesai dengan inference_ms di bawah 1000 di GPU (bandingkan dengan log CPU sebelumnya).
3. `nvidia-smi` saat worker idle menunjukkan pemakaian VRAM stabil (tidak naik terus setiap job, tidak ada memory leak setelah 50 job berturut-turut).
4. Mematikan CUDA secara paksa (env `CUDA_VISIBLE_DEVICES=""`) membuat worker tetap jalan di CPU dengan WARNING, tanpa crash.
5. Job dengan gambar sangat besar (foto 4000 px) tetap berhasil karena resize preprocessing.
6. `benchmark_ocr.py` menghasilkan CSV perbandingan untuk minimal satu engine.
7. Mengganti `OCR_ENGINE=gemini` membuat worker jalan tanpa torch ter-load (verifikasi VRAM 0 dipakai proses worker).

Kerjakan Tugas 9 dulu sampai semua Definition of Done poin 1-5 terpenuhi, baru lanjut Tugas 10.
