import re
from PIL import Image

def extract_receipt_data(image: Image.Image, processor, model, device) -> dict:
    # 1. Proses Gambar
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    decoder_input_ids = decoder_input_ids.to(device)

    # 2. Generasi Output
    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=model.decoder.config.max_position_embeddings,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )

    # 3. Decode & Parse ke JSON
    raw_sequence = processor.batch_decode(outputs.sequences)[0]
    raw_sequence = raw_sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", raw_sequence, count=1).strip()
    
    parsed_data = processor.token2json(sequence)
    print("--- AI RAW OUTPUT ---")
    print(parsed_data)

    # Log to ocr_debug.log inside python_ai_service directory for developer analysis
    import os
    import json
    try:
        log_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ocr_debug.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "="*50 + "\n")
            f.write("=== NEW OCR REQUEST ===\n")
            f.write(f"Raw Sequence: {raw_sequence}\n")
            f.write(f"Parsed JSON: {json.dumps(parsed_data, indent=2)}\n")
    except Exception as e:
        print("Failed to write ocr_debug.log:", e)

    # 4. Normalisasi parsed_data: Donut kadang mengembalikan list, kadang dict
    # Jika list, gabungkan semua dict di dalamnya menjadi satu dict
    if isinstance(parsed_data, list):
        merged = {}
        for item in parsed_data:
            if isinstance(item, dict):
                for k, v in item.items():
                    if k in merged:
                        # Jika key sudah ada, gabungkan value
                        if isinstance(merged[k], list) and isinstance(v, list):
                            merged[k].extend(v)
                        elif isinstance(merged[k], list):
                            merged[k].append(v)
                        else:
                            merged[k] = [merged[k], v]
                    else:
                        merged[k] = v
        parsed_data = merged
        print("--- MERGED parsed_data from list ---")
        print(parsed_data)

    # 5. Ekstraksi Data dengan Fallback
    def get_value(data, keys, default=None):
        if not isinstance(data, dict): return default
        for key in keys:
            if key in data:
                val = data[key]
                return val[0] if isinstance(val, list) and len(val) > 0 else val
        return default

    def clean_price(raw_price):
        """Bersihkan string harga menjadi float. Contoh: '37.000' -> 37000.0"""
        if not raw_price:
            return 0.0
        try:
            price_str = str(raw_price)
            # Hapus semua karakter kecuali digit
            clean = re.sub(r'[^\d]', '', price_str)
            # Jika string asli berakhiran ',00' atau '.00', hapus 2 digit terakhir
            if price_str.endswith(',00') or price_str.endswith('.00'):
                clean = clean[:-2]
            if clean:
                return float(clean)
        except:
            pass
        return 0.0

    merchant_name = None
    total_amount = 0.0
    date = None

    if isinstance(parsed_data, dict):
        # 1. Cari Nama Merchant di store_info
        store_info = parsed_data.get("store_info", {})
        merchant_name = get_value(store_info, ["name", "nm", "store_name"])

        # 2. Cari Tanggal di payment_info
        payment_info = parsed_data.get("payment_info", {})
        date = get_value(payment_info, ["date", "dt"])

        # 3. Cari di menu items untuk merchant name (item pertama sering = nama toko)
        menu_items = parsed_data.get("menu", [])
        if isinstance(menu_items, list) and len(menu_items) > 0:
            if not merchant_name:
                first_item = menu_items[0] if isinstance(menu_items[0], dict) else {}
                first_item_name = str(first_item.get("nm", ""))
                if first_item_name and not re.search(r'\d{4}-\d{2}-\d{2}', first_item_name):
                    merchant_name = first_item_name.strip()

            if not date:
                date_pattern = r'\b(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|\d{2}\.\d{2}\.\d{2,4})\b'
                for item in menu_items:
                    if isinstance(item, dict):
                        for key, val in item.items():
                            match = re.search(date_pattern, str(val))
                            if match:
                                date = match.group(1)
                                break
                    if date:
                        break

        # 4. Cari Total - cek BANYAK kemungkinan lokasi
        # 4a. Cek di root level (subtotal_price, total_price, dll)
        raw_total = None
        for total_key in ["total_price", "subtotal_price", "sub_total_price", "total", "grandtotal_price"]:
            val = parsed_data.get(total_key)
            if val:
                raw_total = val
                break

        # 4b. Cek di dalam section "total"
        if not raw_total:
            total_section = parsed_data.get("total", {})
            raw_total = get_value(total_section, ["total_price", "total", "subtotal_price"])

        # 4c. Cek di payment_info
        if not raw_total:
            raw_total = get_value(payment_info, ["total_price", "total"])

        # 4d. Jika masih kosong, hitung dari semua price di menu items
        if not raw_total and isinstance(menu_items, list):
            calculated_total = 0.0
            for item in menu_items:
                if isinstance(item, dict):
                    item_price = item.get("price", "")
                    if item_price and isinstance(item_price, str) and re.search(r'\d', item_price):
                        # Hanya ambil jika terlihat seperti angka harga
                        price_val = clean_price(item_price)
                        if price_val > 0 and price_val < 100000000:  # sanity check
                            calculated_total += price_val
            if calculated_total > 0:
                raw_total = str(int(calculated_total))

        # 5. Bersihkan Angka Total
        if raw_total:
            total_amount = clean_price(raw_total)

    # ================= REGEX FALLBACK SYSTEM =================
    # Jika merchant_name masih kosong, coba regex dari raw_sequence
    if not merchant_name:
        merchant_name = (
            re.search(r'<s_store_info>.*?<s_name>(.*?)</s_name>', raw_sequence, re.IGNORECASE) or
            re.search(r'<s_store_info>.*?<s_nm>(.*?)</s_nm>', raw_sequence, re.IGNORECASE) or
            re.search(r'<s_name>(.*?)</s_name>', raw_sequence, re.IGNORECASE) or
            re.search(r'<s_nm>(.*?)</s_nm>', raw_sequence, re.IGNORECASE)
        )
        merchant_name = merchant_name.group(1).strip() if merchant_name else None

    # Jika date masih kosong
    if not date:
        date_match = (
            re.search(r'<s_date>(.*?)</s_date>', raw_sequence, re.IGNORECASE) or
            re.search(r'<s_dt>(.*?)</s_dt>', raw_sequence, re.IGNORECASE) or
            re.search(r'\b(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|\d{2}\.\d{2}\.\d{2,4})\b', raw_sequence)
        )
        date = date_match.group(1).strip() if date_match else None

    # Jika total_amount masih 0.0, cari di raw_sequence
    if total_amount == 0.0:
        total_match = (
            re.search(r'<s_total_price>(.*?)</s_total_price>', raw_sequence, re.IGNORECASE) or
            re.search(r'<s_subtotal_price>(.*?)</s_subtotal_price>', raw_sequence, re.IGNORECASE) or
            re.search(r'<s_total>(.*?)</s_total>', raw_sequence, re.IGNORECASE) or
            re.search(r'<s_cashprice>(.*?)</s_cashprice>', raw_sequence, re.IGNORECASE) or
            re.search(r'<s_sub_total>(.*?)</s_sub_total>', raw_sequence, re.IGNORECASE)
        )
        if total_match:
            total_amount = clean_price(total_match.group(1))

    # Last resort: cari kata "total" diikuti angka di plaintext
    if total_amount == 0.0:
        plain_sequence = re.sub(r'<.*?>', ' ', raw_sequence)
        for pattern in [
            r'total.*?(\d{1,3}[\.,]\d{3}(?:[\.,]\d{3})*)',
            r'total.*?(\d{4,9})',
        ]:
            match = re.search(pattern, plain_sequence, re.IGNORECASE)
            if match:
                total_amount = clean_price(match.group(1))
                if total_amount > 0:
                    break

    print(f"--- FINAL EXTRACTION: merchant={merchant_name}, total={total_amount}, date={date} ---")
            
    return {
        "status": "success",
        "data": {
            "merchant_name": merchant_name,
            "total_amount": total_amount,
            "date": date,
            "suggested_category": 10
        },
        "debug_raw_ai": parsed_data
    }

