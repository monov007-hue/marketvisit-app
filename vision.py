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

# ══════════════════════════════════════
# CONFIG
# ══════════════════════════════════════

VISION_URL     = "https://vision.googleapis.com/v1/images:annotate"
SERPER_URL     = "https://google.serper.dev/search"
GROQ_MODEL     = "llama-3.3-70b-versatile"
CACHE_DB       = "cache.db"
MAX_IMAGE_SIZE = 1280      # увеличено для лучшего распознавания
JPEG_QUALITY   = 85
SERPER_TIMEOUT = 5.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("vision-groq")

# ══════════════════════════════════════
# CATEGORIES
# ══════════════════════════════════════

CATEGORIES = [
    "Овощи",                      "Фрукты",                       "Зелень и салаты",
    "Грибы",                      "Молоко и сливки",               "Кефир, кисломолочные изделия",
    "Сметана, творог",             "Творог, творожные десерты",     "Сыры",
    "Масло, маргарин",             "Молочные напитки",              "Мороженое",
    "Свинина",                    "Тушка птицы",                  "Разделка куриная",
    "Субпродукты",                "Другие виды мяса",             "Мясные полуфабрикаты",
    "Мясные изделия",             "Колбасные изделия",            "Рыба",
    "Морепродукты",                "Рыба готовая",                "Икра",
    "Консервы мясные",            "Консервы рыбные",              "Хлеб и батоны",
    "Булочки и пирожки",          "Торты и пирожные",             "Печенье и вафли",
    "Сушки и сухари",             "Шоколад и батончики",          "Конфеты и карамель",
    "Зефир и мармелад",           "Халва и козинаки",             "Жевательные резинки",
    "Чипсы и сухарики",           "Орехи и семечки",              "Попкорн",
    "Сухофрукты",                 "Батончики злаковые",           "Крупы",
    "Макароны и лапша",           "Мука и смеси",                 "Бобовые",
    "Растительные масла",         "Майонез и кетчуп",             "Соусы и приправы",
    "Специи и пряности",          "Варенье и джемы",              "Мёд",
    "Сгущённое молоко",           "Сиропы и топпинги",            "Замороженные овощи",
    "Пельмени и вареники",        "Замороженные блюда",           "Овощные консервы",
    "Готовые супы",               "Готовые каши",                 "Паштеты",
    "Детское питание",            "Газированные напитки",         "Соки и нектары",
    "Вода",                       "Энергетики",                   "Чай",
    "Кофе",                       "Алкоголь",                     "Безглютеновые продукты",
    "Диетические продукты",       "Органические продукты",        "Спортивное питание",
    "Растительные альтернативы",  "Другое",
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

    # Сохраняем детали — уменьшаем только если реально большое
    img.thumbnail((1920, 1920))

    # Подбираем качество чтобы уложиться в 7 МБ
    out = io.BytesIO()
    for quality in [85, 75, 65, 55]:
        out = io.BytesIO()
        img.save(out, format="JPEG", quality=quality, optimize=True)
        size_mb = len(out.getvalue()) / (1024 * 1024)
        if size_mb < 7.0:
            logger.info(f"[IMAGE] {size_mb:.1f} МБ, quality={quality}")
            break

    return out.getvalue()

# ══════════════════════════════════════
# CACHE
# ══════════════════════════════════════

def image_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


async def get_cache(h: str):
    async with aiosqlite.connect(CACHE_DB) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS image_cache (
                hash TEXT PRIMARY KEY,
                result TEXT,
                timestamp INTEGER
            )
        """)
        await db.commit()
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
# GOOGLE VISION — 15 результатов
# ══════════════════════════════════════

async def run_vision(image_b64: str) -> str:
    payload = {
        "requests": [{
            "image": {"content": image_b64},
            "features": [
                {"type": "LABEL_DETECTION",       "maxResults": 15},  # было 8
                {"type": "TEXT_DETECTION"},
                {"type": "LOGO_DETECTION",         "maxResults": 5},   # было 3
                {"type": "OBJECT_LOCALIZATION",    "maxResults": 10},  # возвращён
            ],
        }]
    }

    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{VISION_URL}?key={os.getenv('GOOGLE_API_KEY')}",
            json=payload,
        )

    data    = r.json()["responses"][0]
    labels  = [x.get("description", "") for x in data.get("labelAnnotations",          [])]
    logos   = [x.get("description", "") for x in data.get("logoAnnotations",           [])]
    objects = [x.get("name",        "") for x in data.get("localizedObjectAnnotations", [])]
    text    = ""
    if "textAnnotations" in data:
        text = data["textAnnotations"][0].get("description", "")[:700]

    return (
        f"LABELS: {', '.join(labels)}\n"
        f"LOGOS: {', '.join(logos)}\n"
        f"OBJECTS: {', '.join(objects)}\n"
        f"TEXT: {text}"
    ).strip()

# ══════════════════════════════════════
# SERPER — Google Search API
# ══════════════════════════════════════

def build_search_query(vision_text: str) -> str:
    """Строим умный поисковый запрос из данных Vision"""
    logos  = []
    text   = ""
    labels = []

    for line in vision_text.split("\n"):
        if line.startswith("LOGOS:"):
            logos = [x.strip() for x in line.replace("LOGOS:", "").split(",") if x.strip()]
        elif line.startswith("TEXT:"):
            text = line.replace("TEXT:", "").replace("\n", " ").strip()[:120]
        elif line.startswith("LABELS:"):
            labels = [x.strip() for x in line.replace("LABELS:", "").split(",") if x.strip()]

    parts = []
    if logos:
        parts.extend(logos[:2])
    if text:
        parts.append(text)
    if labels and not logos:
        parts.extend(labels[:3])

    return " ".join(parts)[:250]


async def serper_search(query: str) -> str:
    """Поиск через Serper API (Google Search)"""
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key or not query.strip():
        logger.warning("[SERPER] ключ не задан или пустой запрос")
        return ""

    try:
        async with httpx.AsyncClient(timeout=SERPER_TIMEOUT) as client:
            r = await client.post(
                SERPER_URL,
                headers={
                    "X-API-KEY":    api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": 5, "gl": "ru", "hl": "ru"},
            )

        data     = r.json()
        snippets = []

        for item in data.get("organic", [])[:5]:
            title   = item.get("title",   "")
            snippet = item.get("snippet", "")
            if snippet:
                snippets.append(f"{title}: {snippet}")

        result = "\n".join(snippets)
        logger.info(f"[SERPER] найдено {len(snippets)} результатов")
        return result

    except asyncio.TimeoutError:
        logger.warning("[SERPER] таймаут — пропускаем")
        return ""
    except Exception as e:
        logger.warning(f"[SERPER] ошибка: {e}")
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

    web_block = f"\nWEB (результаты Google):\n{web_info}" if web_info else ""

    prompt = f"""Ты эксперт по товарам супермаркета. Определи товар по данным компьютерного зрения.

VISION API:
{vision_text}{web_block}

Верни только JSON без пояснений:
{{
  "product_name": "точное название с объёмом/весом если есть",
  "brand": "бренд или производитель",
  "category": "одна категория из списка"
}}

Категории (выбери ТОЧНО одну):
{CATEGORIES_STR}

Правила:
- Текст с упаковки (TEXT) — главный источник
- Логотипы (LOGOS) — это бренд
- Если есть объём/вес в тексте — включи в название
- Если не уверен → product_name: "Не опознано", category: "Другое"
"""

    resp = await client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200,
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
    # 1. Сжимаем и улучшаем изображение
    image_bytes = preprocess_image(image_bytes)
    h = image_hash(image_bytes)

    # 2. Проверяем кэш
    cached = await get_cache(h)
    if cached:
        logger.info("[CACHE] hit")
        return cached

    image_b64 = base64.b64encode(image_bytes).decode()

    # 3. Vision API
    logger.info("[Pipeline] Vision API...")
    vision_data = await run_vision(image_b64)

    # 4. Serper — умный запрос на основе Vision
    web_info = ""
    query    = build_search_query(vision_data)
    if query:
        logger.info(f"[Pipeline] Serper: {query[:80]}")
        web_info = await serper_search(query)

    # 5. Groq структурирует результат
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
