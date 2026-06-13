import re
import time
from PIL import Image
import torch

def clean_price(raw_price):
    if not raw_price:
        return 0.0
    try:
        price_str = str(raw_price).strip()
        amount_part = re.sub(r'[^\d\.,]', '', price_str)
        if not amount_part:
            return 0.0

        # Indonesian receipts usually use "." or "," as thousand separators
        # (43.000 => 43000, 93.200 => 93200). If a real decimal suffix exists
        # after a larger integer value (43000.00 or 43.000,00), remove only
        # that cents suffix. Keep OCR-ish values like 932.00 as 93200 because
        # they often originate from 93.200.
        last_dot = amount_part.rfind(".")
        last_comma = amount_part.rfind(",")
        last_sep = max(last_dot, last_comma)

        if last_sep >= 0:
            before = amount_part[:last_sep]
            after = amount_part[last_sep + 1:]
            before_digits = re.sub(r'\D', '', before)
            has_other_separator = any(sep in before for sep in [".", ","])

            if len(after) == 2 and (has_other_separator or len(before_digits) > 3):
                amount_part = before

        clean = re.sub(r'\D', '', amount_part)
        if not clean:
            return 0.0

        return float(clean)
    except:
        pass
    return 0.0

def extract_receipt_data(image: Image.Image, processor, model, device) -> tuple[dict, dict]:
    timings = {}
    
    # 1. Proses Gambar
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device, dtype=model.dtype)

    task_prompt = "<s_cord-v2>"
    decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
    decoder_input_ids = decoder_input_ids.to(device)

    # 2. Generasi Output (Inference Mode)
    t_inf = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=512,  # Batasi secukupnya untuk struk agar lebih cepat
            num_beams=1,     # Greedy decoding
            use_cache=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
            return_dict_in_generate=True,
        )
    timings['inference_ms'] = (time.time() - t_inf) * 1000

    # 3. Decode & Parse ke JSON
    t_parse = time.time()
    raw_sequence = processor.batch_decode(outputs.sequences)[0]
    raw_sequence = raw_sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    sequence = re.sub(r"<.*?>", "", raw_sequence, count=1).strip()
    
    parsed_data = processor.token2json(sequence)

    # 4. Normalisasi parsed_data: Donut kadang mengembalikan list, kadang dict
    if isinstance(parsed_data, list):
        merged = {}
        for item in parsed_data:
            if isinstance(item, dict):
                for k, v in item.items():
                    if k in merged:
                        if isinstance(merged[k], list) and isinstance(v, list):
                            merged[k].extend(v)
                        elif isinstance(merged[k], list):
                            merged[k].append(v)
                        else:
                            merged[k] = [merged[k], v]
                    else:
                        merged[k] = v
        parsed_data = merged

    # 5. Ekstraksi Data dengan Fallback
    def get_value(data, keys, default=None):
        if not isinstance(data, dict): return default
        for key in keys:
            if key in data:
                val = data[key]
                return val[0] if isinstance(val, list) and len(val) > 0 else val
        return default

    merchant_name = None
    total_amount = 0.0
    date = None

    if isinstance(parsed_data, dict):
        store_info = parsed_data.get("store_info", {})
        merchant_name = get_value(store_info, ["name", "nm", "store_name"])

        payment_info = parsed_data.get("payment_info", {})
        date = get_value(payment_info, ["date", "dt"])

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

        raw_total = None
        for total_key in ["total_price", "subtotal_price", "sub_total_price", "total", "grandtotal_price"]:
            val = parsed_data.get(total_key)
            if val:
                raw_total = val
                break

        if not raw_total:
            total_section = parsed_data.get("total", {})
            raw_total = get_value(total_section, ["total_price", "total", "subtotal_price"])

        if not raw_total:
            raw_total = get_value(payment_info, ["total_price", "total"])

        if not raw_total and isinstance(menu_items, list):
            calculated_total = 0.0
            for item in menu_items:
                if isinstance(item, dict):
                    item_price = item.get("price", "")
                    if item_price and isinstance(item_price, str) and re.search(r'\d', item_price):
                        price_val = clean_price(item_price)
                        if price_val > 0 and price_val < 100000000:
                            calculated_total += price_val
            if calculated_total > 0:
                raw_total = str(int(calculated_total))

        if raw_total:
            total_amount = clean_price(raw_total)

    if not merchant_name:
        merchant_name_match = (
            re.search(r'<s_store_info>.*?<s_name>(.*?)</s_name>', raw_sequence, re.IGNORECASE) or
            re.search(r'<s_store_info>.*?<s_nm>(.*?)</s_nm>', raw_sequence, re.IGNORECASE) or
            re.search(r'<s_name>(.*?)</s_name>', raw_sequence, re.IGNORECASE) or
            re.search(r'<s_nm>(.*?)</s_nm>', raw_sequence, re.IGNORECASE)
        )
        merchant_name = merchant_name_match.group(1).strip() if merchant_name_match else None

    if not date:
        date_match = (
            re.search(r'<s_date>(.*?)</s_date>', raw_sequence, re.IGNORECASE) or
            re.search(r'<s_dt>(.*?)</s_dt>', raw_sequence, re.IGNORECASE) or
            re.search(r'\b(\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}|\d{2}\.\d{2}\.\d{2,4})\b', raw_sequence)
        )
        date = date_match.group(1).strip() if date_match else None

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

    timings['parsing_ms'] = (time.time() - t_parse) * 1000

    result = {
        "status": "success",
        "data": {
            "merchant_name": merchant_name,
            "total_amount": total_amount,
            "date": date,
            "suggested_category": 10
        },
        "debug_raw_ai": parsed_data
    }
    
    return result, timings
