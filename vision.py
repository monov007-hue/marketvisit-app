import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import re
import time
from typing import Dict, Any

import aiosqlite
import httpx
from PIL import Image, ImageOps, ImageFilter
from rapidfuzz import process, fuzz
from groq import AsyncGroq
SERPER_URL     = "https://google.serper.dev/search"
SERPER_TIMEOUT = 5.0
async def serper_search(query: str) -> str:
    if not query.strip():
        return ""

    try:
        async with httpx.AsyncClient(timeout=SERPER_TIMEOUT) as client:
            r = await client.post(
                SERPER_URL,
                headers={
                    "X-API-KEY": os.getenv("SERPER_API_KEY"),
                    "Content-Type": "application/json",
                },
                json={
                    "q": query,
                    "gl": "ru",
                    "hl": "ru",
                    "num": 5,
                },
            )

        data = r.json()

        results = []

        for item in data.get("organic", []):
            title = item.get("title", "")
            snippet = item.get("snippet", "")

            if title:
                results.append(f"{title}\n{snippet}")

        return "\n\n".join(results)

    except Exception as e:
        logger.warning(f"[SERPER] ошибка: {e}")
        return ""
def build_search_query(vision_text: str) -> str:
    labels = []
    logos = []
    text = ""

    for line in vision_text.split("\n"):
        if line.startswith("LABELS:"):
            labels = [
                x.strip()
                for x in line.replace("LABELS:", "").split(",")
                if x.strip()
            ]

        elif line.startswith("LOGOS:"):
            logos = [
                x.strip()
                for x in line.replace("LOGOS:", "").split(",")
                if x.strip()
            ]

        elif line.startswith("TEXT:"):
            text = line.replace("TEXT:", "").replace("\n", " ").strip()

    parts = []

    if logos:
        parts.extend(logos[:2])

    if text:
        parts.append(text[:120])

    if labels:
        parts.extend(labels[:3])

    query = " ".join(parts)

    return query[:250]

# ══════════════════════════════════════
# CONFIG
# ══════════════════════════════════════

VISION_URL     = "https://vision.googleapis.com/v1/images:annotate"
GROQ_MODEL     = "llama-3.3-70b-versatile"
CACHE_DB       = "cache.db"
MAX_IMAGE_SIZE = 960       # было 1280 — меньше = быстрее
JPEG_QUALITY   = 82        # было 88
DDG_TIMEOUT    = 4.0       # секунд на веб-поиск, потом пропускаем

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("vision-groq")

# ══════════════════════════════════════
# CATEGORIES
# ══════════════════════════════════════

CATEGORIES = [
    "Свежие овощи",          "Свежие фрукты",         "Зелень и салаты",
    "Грибы",                 "Молоко",                 "Кефир и йогурты",
    "Сметана и сливки",      "Творог и творожки",      "Сыры",
    "Масло сливочное",       "Молочные напитки",       "Мороженое",
    "Мясо",                  "Колбасы и сосиски",      "Рыба и морепродукты",
    "Консервы мясные",       "Консервы рыбные",        "Хлеб и батоны",
    "Булочки и пирожки",     "Торты и пирожные",       "Печенье и вафли",
    "Сушки и сухари",        "Шоколад и батончики",    "Конфеты и карамель",
    "Зефир и мармелад",      "Халва и козинаки",       "Жевательные резинки",
    "Чипсы и сухарики",      "Орехи и семечки",        "Попкорн",
    "Сухофрукты",            "Батончики злаковые",     "Крупы",
    "Макароны и лапша",      "Мука и смеси",           "Бобовые",
    "Растительные масла",    "Майонез и кетчуп",       "Соусы и приправы",
    "Специи и пряности",     "Варенье и джемы",        "Мёд",
    "Сгущённое молоко",      "Сиропы и топпинги",      "Замороженные овощи",
    "Пельмени и вареники",   "Замороженные блюда",     "Овощные консервы",
    "Готовые супы",          "Готовые каши",           "Паштеты",
    "Детское питание",       "Газированные напитки",   "Соки и нектары",
    "Вода",                  "Энергетики",             "Чай",
    "Кофе",                  "Алкоголь",               "Безглютеновые продукты",
    "Диетические продукты",  "Органические продукты",  "Спортивное питание",
    "Растительные альтернативы", "Другое",
]

CATEGORIES_STR = ", ".join(CATEGORIES)

# ══════════════════════════════════════
# IMAGE PREPROCESS
# ══════════════════════════════════════

def preprocess_image(image_bytes: bytes) -> bytes:
    img = Image.open(io.BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)

    if img.mode != "RGB":
        img = img.convert("RGB")

    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    img.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE))

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return out.getvalue()

# ══════════════════════════════════════
# CACHE
# ══════════════════════════════════════

def image_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


async def get_cache(h: str):
    async with aiosqlite.connect(CACHE_DB) as db:
        async with db.execute(
            "SELECT result FROM image_cache WHERE hash=?", (h,)
        ) as cur:
            row = await cur.fetchone()
    return json.loads(row[0]) if row else None


async def save_cache(h: str, result: dict):
    async with aiosqlite.connect(CACHE_DB) as db:
        await db.execute(
            "INSERT OR REPLACE INTO image_cache VALUES (?, ?, ?)",
            (h, json.dumps(result, ensure_ascii=False), int(time.time()))
        )
        await db.commit()

# ══════════════════════════════════════
# GOOGLE VISION
# ══════════════════════════════════════

async def run_vision(image_b64: str) -> str:
    payload = {
        "requests": [{
            "image": {"content": image_b64},
            "features": [
                {"type": "LABEL_DETECTION",  "maxResults": 8},
                {"type": "TEXT_DETECTION"},
                {"type": "LOGO_DETECTION",   "maxResults": 3},
                # OBJECT_LOCALIZATION убран — дублирует LABEL и замедляет
            ],
        }]
    }

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{VISION_URL}?key={os.getenv('GOOGLE_API_KEY')}",
            json=payload,
        )

    data   = r.json()["responses"][0]
    labels = [x.get("description", "") for x in data.get("labelAnnotations", [])]
    logos  = [x.get("description", "") for x in data.get("logoAnnotations",  [])]
    text   = ""
    if "textAnnotations" in data:
        text = data["textAnnotations"][0].get("description", "")[:500]  # обрезаем длинный текст

    return f"LABELS: {', '.join(labels)}\nLOGOS: {', '.join(logos)}\nTEXT: {text}".strip()

# ══════════════════════════════════════
# DDG ПОИСК — async с таймаутом
# ══════════════════════════════════════

def _ddg_sync(query: str) -> str:
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=3)
            return "\n".join(r["body"] for r in results)
    except:
        return ""


async def ddg_search_async(query: str) -> str:
    """Запускает DDG в отдельном потоке с таймаутом — не блокирует event loop"""
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(_ddg_sync, query),
            timeout=DDG_TIMEOUT
        )
        return result
    except asyncio.TimeoutError:
        logger.warning("[DDG] таймаут — пропускаем веб-поиск")
        return ""
    except Exception as e:
        logger.warning(f"[DDG] ошибка: {e}")
        return ""

# ══════════════════════════════════════
# GROQ
# ══════════════════════════════════════

def safe_json_load(text: str) -> dict:
    text = text.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    return json.loads(text)


async def run_groq(vision_text: str, web_info: str) -> dict:
    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

    web_block = f"\nWEB:\n{web_info}" if web_info else ""

    prompt = f"""Ты эксперт по товарам супермаркета.

VISION:
{vision_text}{web_block}

Верни только JSON без пояснений:
{{
  "product_name": "...",
  "brand": "...",
  "category": "..."
}}

Категории: {CATEGORIES_STR}
Если не уверен → "Не опознано" / "Другое"
"""

    resp = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200,   # нам нужен только короткий JSON
    )

    content = resp.choices[0].message.content
    logger.info(f"[GROQ] {content}")

    try:
        return safe_json_load(content)
    except:
        return {"product_name": "Не опознано", "brand": "Не опознано", "category": "Другое"}

# ══════════════════════════════════════
# CATEGORY NORMALIZE
# ══════════════════════════════════════

def normalize_category(cat: str) -> str:
    best = process.extractOne(cat, CATEGORIES, scorer=fuzz.token_sort_ratio)
    return best[0] if best and best[1] > 60 else "Другое"

# ══════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════

async def analyze_image(image_bytes: bytes) -> Dict[str, Any]:
    # 1. Сжимаем изображение
    image_bytes = preprocess_image(image_bytes)
    h = image_hash(image_bytes)

    # 2. Проверяем кэш
    cached = await get_cache(h)
    if cached:
        logger.info("[CACHE] hit")
        return cached

    image_b64 = base64.b64encode(image_bytes).decode()

    # 3. Vision API и DDG запускаем ПАРАЛЛЕЛЬНО
    logger.info("[Pipeline] Vision + DDG параллельно...")
    vision_data, web_info = await asyncio.gather(
        run_vision(image_b64),
        ddg_search_async(image_b64[:100]),  # DDG стартует сразу, пока Vision обрабатывает
    )

    # DDG с нормальным запросом на основе Vision результата
    # Если Vision быстрее DDG — делаем уточняющий поиск по результату
    if not web_info and vision_data:
        query = vision_data.split("\n")[1] if "\n" in vision_data else vision_data[:80]
        web_info = await ddg_search_async(query)

    # 4. Groq структурирует результат
    logger.info("[Groq] structuring...")
    result = await run_groq(vision_data, web_info)

    result["category"] = normalize_category(result.get("category", ""))

    final = {
        "recognized":       result.get("product_name") != "Не опознано",
        "product_name":     result.get("product_name",     "Не опознано"),
        "brand":            result.get("brand",            "Не опознано"),
        "product_category": result["category"],
        "confidence":       1.0,
    }

    await save_cache(h, final)
    return final
