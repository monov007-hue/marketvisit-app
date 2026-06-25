import os
import uuid
import json
import base64
import logging
import asyncio
import aiosqlite
import httpx

from aiohttp import web
from dotenv import load_dotenv

from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
    WebAppInfo,
)
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from database import (
    init_db,
    save_product,
    update_product,
    delete_product,
    save_group_photo,
    get_recent_group_photos,
    save_feedback,
)
from vision import analyze_image

# ══════════════════════════════════════
# CONFIG
# ══════════════════════════════════════

load_dotenv()

BOT_TOKEN    = os.getenv("BOT_TOKEN")
BOT_USERNAME = os.getenv("BOT_USERNAME")
WEBAPP_URL   = os.getenv("WEBAPP_URL")
PHOTOS_DIR   = "photos"

os.makedirs(PHOTOS_DIR, exist_ok=True)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("market-bot")

# ══════════════════════════════════════
# MEMORY
# ══════════════════════════════════════

user_states: dict = {}
bot_instance      = None

# ══════════════════════════════════════
# DB INIT
# ══════════════════════════════════════

async def post_init(app: Application):
    global bot_instance
    bot_instance = app.bot
    await init_db()
    # Создаём кэш таблицу для vision.py
    async with aiosqlite.connect("cache.db") as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS image_cache (
                hash TEXT PRIMARY KEY,
                result TEXT,
                timestamp INTEGER
            )
        """)
        await db.commit()

# ══════════════════════════════════════
# KEYBOARDS
# ══════════════════════════════════════

def keyboard_private():
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(
            text="📸 Открыть загрузку",
            web_app=WebAppInfo(url=WEBAPP_URL)
        )
    ]])


def keyboard_group():
    return InlineKeyboardMarkup([[
        InlineKeyboardButton(
            text="📸 Фото",
            url=f"https://t.me/{BOT_USERNAME}/upload"
        )
    ]])


def keyboard_result(row_id: int):
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("✅ Отправить", callback_data=f"done_{row_id}"),
        InlineKeyboardButton("✏️ Изменить",  callback_data=f"edit_{row_id}"),
        InlineKeyboardButton("❌ Удалить",   callback_data=f"del_{row_id}"),
    ]])

# ══════════════════════════════════════
# COMMANDS
# ══════════════════════════════════════

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📸 Открой мини-приложение для загрузки фото",
        reply_markup=keyboard_private()
    )


async def pin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat = update.effective_chat

    if chat.type == chat.PRIVATE:
        await update.message.reply_text(
            "НАЖМИ ДЛЯ АНАЛИЗА 👉👉",
            reply_markup=keyboard_private()
        )
        return

    try:
        msg = await update.message.reply_text(
            "НАЖМИ ДЛЯ АНАЛИЗА 👉👉",
            reply_markup=keyboard_group()
        )
        await context.bot.pin_chat_message(
            chat_id=chat.id,
            message_id=msg.message_id,
            disable_notification=True,
        )
    except Exception as e:
        logger.error(f"[PIN ERROR] {e}")
        await update.message.reply_text(f"⚠️ Ошибка закрепления: {e}")

# ══════════════════════════════════════
# PHOTO HANDLERS
# ══════════════════════════════════════

async def handle_group_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        photo = update.message.photo[-1]
        await save_group_photo(
            file_id        = photo.file_id,
            file_unique_id = photo.file_unique_id,
            chat_id        = update.effective_chat.id,
        )
        logger.info(f"[GROUP PHOTO] сохранён file_id: {photo.file_id}")
    except Exception as e:
        logger.error(f"[GROUP PHOTO ERROR] {e}")


async def handle_private_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user  = update.effective_user
    photo = update.message.photo[-1]
    file  = await context.bot.get_file(photo.file_id)

    image_bytes = await file.download_as_bytearray()
    path        = os.path.join(PHOTOS_DIR, f"{uuid.uuid4()}.jpg")

    with open(path, "wb") as f:
        f.write(image_bytes)

    result = await analyze_image(bytes(image_bytes))

    row_id = await save_product(
        user_id          = user.id,
        username         = user.username or str(user.id),
        category         = "design",
        photo_path       = path,
        product_name     = result.get("product_name",     "Не опознано"),
        brand            = result.get("brand",            "Не опознано"),
        product_category = result.get("product_category", "Другое"),
        confidence       = result.get("confidence",       0),
    )

    user_states[user.id] = {"row_id": row_id, "photo_path": path}

    await update.message.reply_text(
        f"📦 Название:  {result.get('product_name',     '—')}\n"
        f"🏷 Бренд:     {result.get('brand',            '—')}\n"
        f"🗂 Категория: {result.get('product_category', '—')}",
        reply_markup=keyboard_result(row_id)
    )

# ══════════════════════════════════════
# API — /api/analyze
# Принимает base64 фото от Cloudflare Worker,
# прогоняет через vision.py (Vision+DDG+Groq),
# возвращает JSON с результатом
# ══════════════════════════════════════

async def analyze_api(request):
    """POST /api/analyze — вызывается Cloudflare Worker"""
    # CORS preflight
    if request.method == "OPTIONS":
        return web.Response(
            status=204,
            headers={
                "Access-Control-Allow-Origin":  "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )

    try:
        body        = await request.json()
        image_b64   = body.get("image", "")

        if not image_b64:
            return web.json_response(
                {"error": "Поле image обязательно"},
                status=400,
                headers={"Access-Control-Allow-Origin": "*"}
            )

        # Декодируем base64 → bytes
        image_bytes = base64.b64decode(image_b64)

        # Полный pipeline: Vision + DDG + Groq + rapidfuzz
        logger.info("[ANALYZE API] запуск анализа...")
        result = await analyze_image(image_bytes)
        logger.info(f"[ANALYZE API] результат: {result}")

        # Сохраняем фото на диск
        path = os.path.join(PHOTOS_DIR, f"{uuid.uuid4()}.jpg")
        with open(path, "wb") as f:
            f.write(image_bytes)

        # Сохраняем в БД
        row_id = await save_product(
            user_id          = 0,
            username         = "webapp",
            category         = "webapp",
            photo_path       = path,
            product_name     = result.get("product_name",     "Не опознано"),
            brand            = result.get("brand",            "Не опознано"),
            product_category = result.get("product_category", "Другое"),
            confidence       = result.get("confidence",       0),
        )

        return web.json_response(
            {
                "success":      True,
                "product_id":   row_id, 
                "product_name": result.get("product_name",     "Не опознано"),
                "brand":        result.get("brand",            "Не опознано"),
                "category":     result.get("product_category", "Другое"),
                "confidence":   result.get("confidence",       1.0),
            },
            headers={"Access-Control-Allow-Origin": "*"}
        )

    except Exception as e:
        logger.error(f"[ANALYZE API ERROR] {e}")
        return web.json_response(
            {"error": str(e)},
            status=500,
            headers={"Access-Control-Allow-Origin": "*"}
        )


# ══════════════════════════════════════
# API — /api/photos
# ══════════════════════════════════════

async def photos_api(request):
    try:
        file_ids = await get_recent_group_photos(limit=12)
        photos   = []

        for file_id in file_ids:
            try:
                tg_file = await bot_instance.get_file(file_id)
                file_path = tg_file.file_path
                if file_path.startswith("http"):
                    url = file_path
                else:
                    url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_path}"
                photos.append({"url": url})
            except Exception as e:
                logger.warning(f"[PHOTOS API] file_id {file_id}: {e}")

        return web.json_response(
            {"photos": photos},
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        logger.error(f"[PHOTOS API] {e}")
        return web.json_response({"photos": []})

async def feedback_api(request):
    if request.method == "OPTIONS":
        return web.Response(
            status=204,
            headers={
                "Access-Control-Allow-Origin":  "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )
    try:
        body       = await request.json()
        product_id = body.get("product_id")
        vote       = body.get("vote")

        if product_id is None or vote not in (1, -1):
            return web.json_response(
                {"error": "product_id и vote (1 или -1) обязательны"},
                status=400,
                headers={"Access-Control-Allow-Origin": "*"}
            )

        await save_feedback(
            product_id       = product_id,
            vote             = vote,
            correct_name     = body.get("correct_name"),
            correct_brand    = body.get("correct_brand"),
            correct_category = body.get("correct_category"),
        )

        logger.info(f"[FEEDBACK] product_id={product_id} vote={vote}")
        return web.json_response(
            {"success": True},
            headers={"Access-Control-Allow-Origin": "*"}
        )

    except Exception as e:
        logger.error(f"[FEEDBACK ERROR] {e}")
        return web.json_response(
            {"error": str(e)},
            status=500,
            headers={"Access-Control-Allow-Origin": "*"}
        )
        
async def proxy_photo(request):
    """GET /api/proxy?url=... — проксирует фото из Telegram"""
    url = request.query.get("url")
    if not url or "api.telegram.org" not in url:
        return web.Response(status=400)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
        return web.Response(
            body=r.content,
            content_type=r.headers.get("content-type", "image/jpeg"),
            headers={"Access-Control-Allow-Origin": "*"}
        )
    except Exception as e:
        return web.Response(status=500)
        
async def start_web_server():
    app  = web.Application()
    app.router.add_get("/api/photos",   photos_api)
    app.router.add_post("/api/analyze", analyze_api)
    app.router.add_route("OPTIONS", "/api/analyze", analyze_api)
    app.router.add_post("/api/feedback", feedback_api)
    app.router.add_route("OPTIONS", "/api/feedback", feedback_api)
    app.router.add_get("/api/proxy", proxy_photo)

    runner = web.AppRunner(app)
    await runner.setup()

    port = int(os.getenv("PORT", 8080))
    site = web.TCPSite(runner, "0.0.0.0", port)
    await site.start()
    logger.info(f"✅ API сервер запущен на порту {port}")

# ══════════════════════════════════════
# WEBAPP DATA
# ══════════════════════════════════════

async def handle_webapp_data(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        payload     = json.loads(update.message.web_app_data.data)
        image_bytes = base64.b64decode(payload["image"])
        chat_id     = payload.get("chat_id")

        result = await analyze_image(image_bytes)

        filename = f"{uuid.uuid4()}.jpg"
        path     = os.path.join(PHOTOS_DIR, filename)
        with open(path, "wb") as f:
            f.write(image_bytes)

        await update.message.reply_text(
            f"📦 {result.get('product_name',     '—')}\n"
            f"🏷 {result.get('brand',            '—')}\n"
            f"🗂 {result.get('product_category', '—')}"
        )

        if chat_id:
            try:
                with open(path, "rb") as f:
                    await context.bot.send_photo(chat_id=int(chat_id), photo=f)
                logger.info(f"[WEBAPP] Фото отправлено в группу {chat_id}")
            except Exception as e:
                logger.warning(f"[WEBAPP] Не удалось отправить в группу: {e}")

    except Exception as e:
        logger.error(f"[WEBAPP ERROR] {e}")
        await update.message.reply_text("⚠️ Не удалось обработать данные из приложения.")

# ══════════════════════════════════════
# INLINE BUTTONS
# ══════════════════════════════════════

async def handle_buttons(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query   = update.callback_query
    await query.answer()

    data    = query.data
    user_id = query.from_user.id

    if data.startswith("done_"):
        state = user_states.get(user_id)
        if not state:
            await query.edit_message_text("⚠️ Состояние не найдено.")
            return
        try:
            with open(state["photo_path"], "rb") as f:
                await context.bot.send_photo(chat_id=query.message.chat_id, photo=f)
            await query.edit_message_text("✅ Фото отправлено в чат")
        except FileNotFoundError:
            await query.edit_message_text("⚠️ Файл не найден.")

    elif data.startswith("del_"):
        await delete_product(int(data.split("_")[1]))
        await query.edit_message_text("❌ Удалено")

    elif data.startswith("edit_"):
        if user_id not in user_states:
            user_states[user_id] = {}
        user_states[user_id]["editing"] = int(data.split("_")[1])
        await query.edit_message_text(
            "✏️ Введите через запятую:\nНазвание, Бренд, Категория"
        )

# ══════════════════════════════════════
# EDIT TEXT HANDLER
# ══════════════════════════════════════

async def handle_edit(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    state   = user_states.get(user_id)

    if not state or "editing" not in state:
        return

    parts    = [x.strip() for x in update.message.text.strip().split(",")]
    name     = parts[0] if len(parts) > 0 else "Не опознано"
    brand    = parts[1] if len(parts) > 1 else "Не опознано"
    category = parts[2] if len(parts) > 2 else "Другое"

    await update_product(state["editing"], name, brand, category)
    del state["editing"]

    await update.message.reply_text(
        f"✅ Обновлено\n\n📦 {name}\n🏷 {brand}\n🗂 {category}"
    )

# ══════════════════════════════════════
# ERROR HANDLER
# ══════════════════════════════════════

async def error_handler(update, context):
    logger.exception(context.error)

# ══════════════════════════════════════
# MAIN
# ══════════════════════════════════════

def main():
    app = (
        Application.builder()
        .token(BOT_TOKEN)
        .post_init(post_init)
        .build()
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("pin",   pin))
    app.add_handler(CallbackQueryHandler(handle_buttons))
    app.add_handler(MessageHandler(
        filters.StatusUpdate.WEB_APP_DATA, handle_webapp_data
    ))
    app.add_handler(MessageHandler(
        filters.ChatType.GROUPS & filters.PHOTO, handle_group_photo
    ))
    app.add_handler(MessageHandler(
        filters.ChatType.PRIVATE & filters.PHOTO, handle_private_photo
    ))
    app.add_handler(MessageHandler(
        filters.ChatType.PRIVATE & filters.TEXT & ~filters.COMMAND, handle_edit
    ))
    app.add_error_handler(error_handler)

    async def run():
        await start_web_server()
        await app.initialize()
        await post_init(app)
        
        await app.start()
        await app.updater.start_polling()

        logger.info("✅ Бот запущен")

        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            pass
        finally:
            await app.updater.stop()
            await app.stop()
            await app.shutdown()
            logger.info("🛑 Бот остановлен")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("🛑 Бот остановлен вручную")


if __name__ == "__main__":
    main()
