# Dokumentasi API Service AI (Local OCR)

Service Python ini dibangun menggunakan FastAPI untuk menangani tugas-tugas *Machine Learning* yang berat bagi backend Laravel utama, khususnya fitur **Optical Character Recognition (OCR) Lokal** secara *offline* (dengan Hugging Face Transformers & PyTorch).

## 🚀 Fitur Auto-Scan Kuitansi (OCR Lokal)

Karena isu pemblokiran jaringan ISP ke huggingface inference API, aplikasi kini di-hosting langsung secara lokal dengan memanfaatkan performa akselerasi GPU (CUDA).

### 1. Spesifikasi Model
- **Model Dasar**: `naver-clova-ix/donut-base-finetuned-cord-v2`
- **Arsitektur**: Vision Encoder-Decoder Model (Donut)
- **Library**: PyTorch, Transformers, Pillow, SentencePiece
- **Inisialisasi**: Model dimuat satu kali secara Global Statement ke dalam VRAM (*Video RAM*) GPU melalui fitur `lifespan` bawaan FastAPI saat server `uvicorn` dinyalakan.

### 2. Endpoint OCR
**`POST /predict/ocr`**

Menerima berkas gambar kuitansi, memprosesnya melalui model *Vision-to-Text* milik Clóva, lalu mem-parsing string berbentuk semi-XML menjadi bentuk Data JSON yang terstruktur.

#### **Request**
- **Content-Type**: `multipart/form-data`
- **Parameter**:
  - `receiptImage` (File: jpg/jpeg/png/webp) - Gambar kuitansi yang difoto/diunggah pengguna.

#### **Response Sukses (200 OK)**
```json
{
    "status": "success",
    "data": {
        "merchant_name": "NAMA TOKO / KASIR",
        "total_amount": 150000.0,
        "date": "2026-04-27",
        "suggested_category": "Pengeluaran Lainnya"
    },
    "debug_raw_ai": {
         // Struktur asli dari hasil token2json milik HuggingFace
    }
}
```

#### **Contoh Balasan Gagal (400 / 500)**
```json
{
    "detail": "Model OCR gagal dimuat. Cek log terminal."
}
```

### 3. Logika Ekstraksi Data (Fallback System)
Karena struktur kuitansi bervariasi, parsing diterapkan secara bertingkat (*fallback pipeline*):
1. Algoritma menggunakan pustaka bawaan `processor.token2json()` untuk merapikan karakter `<s_cord-v2>` menjadi format kamus Python (dictionary).
2. **Merchant Name**: Dicari pada blok `store_info` -> `name` / `nm` / `store_name`. Jika tidak ada, ia meminjam nama barang baris pertama pada daftar `menu` (selama bukan kombinasi angka tanggal).
3. **Date**: Dicari pada blok `payment_info` -> `date` / `dt`. Jika kosong, algoritma Regex akan menelusuri keseluruhan area teks `menu` untuk mencari format tanggal standar (`YYYY-MM-DD` atau `DD-MM-YYYY`).
4. **Total Price**: Diambil memprioritaskan blok `total` -> `total_price` lalu mem-filter karakter koma/titik untuk mengubahnya menjadi `float`.


---

## 🛠️ Instalasi & Menjalankan Service

1. Aktifkan virtual environment Anda.
   ```bash
   # Windows (Powershell)
   .\env\Scripts\activate
   ```
2. Pastikan dependensi sudah terinstall (khususnya library besar berukuran ~3 GB).
   ```bash
   pip install -r requirements.txt
   ```
   > **Note**: PyTorch harus versi >= 2.6.x (atau menggunakan fallback parameter safe_tensors)
3. Jalankan server Uvicorn.
   ```bash
   uvicorn main:app --reload --port 8000
   ```
4. Apabila berhasil, terminal akan memunculkan:
   ```text
   Loading Donut OCR model on cuda (Worker Process)...
   Model loaded successfully into the active worker!
   Application startup complete.
   ```
