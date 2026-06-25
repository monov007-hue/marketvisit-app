import aiosqlite

DB_FILE  = "market_visit.db"
CACHE_DB = "cache.db"

async def init_db():
    # Основная база
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                file_path TEXT,
                created_at INTEGER
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                username TEXT,
                category TEXT,
                photo_path TEXT,
                product_name TEXT,
                brand TEXT,
                product_category TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                is_corrected INTEGER DEFAULT 0
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS group_photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id TEXT NOT NULL,
                file_unique_id TEXT UNIQUE NOT NULL,
                chat_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Таблица фидбека — лайки/дизлайки и исправления
        await db.execute("""
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                vote INTEGER NOT NULL,        -- 1 лайк, -1 дизлайк
                correct_name TEXT,            -- исправленное название (если дизлайк)
                correct_brand TEXT,           -- исправленный бренд
                correct_category TEXT,        -- исправленная категория
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (product_id) REFERENCES products(id)
            )
        """)
        await db.commit()

    # Кэш база (отдельный файл)
    async with aiosqlite.connect(CACHE_DB) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS image_cache (
                hash TEXT PRIMARY KEY,
                result TEXT,
                timestamp INTEGER
            )
        """)
        await db.commit()

# ══════════════════════════════════════
# PRODUCTS
# ══════════════════════════════════════

async def save_product(user_id, username, category, photo_path,
                       product_name, brand, product_category, confidence):
    async with aiosqlite.connect(DB_FILE) as db:
        cursor = await db.execute("""
            INSERT INTO products
            (user_id, username, category, photo_path,
             product_name, brand, product_category, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, username, category, photo_path,
              product_name, brand, product_category, confidence))
        await db.commit()
        return cursor.lastrowid


async def update_product(row_id, product_name, brand, product_category):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
            UPDATE products
            SET product_name=?, brand=?, product_category=?, is_corrected=1
            WHERE id=?
        """, (product_name, brand, product_category, row_id))
        await db.commit()


async def delete_product(row_id):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("DELETE FROM products WHERE id=?", (row_id,))
        await db.commit()

# ══════════════════════════════════════
# FEEDBACK
# ══════════════════════════════════════

async def save_feedback(product_id: int, vote: int,
                        correct_name: str = None,
                        correct_brand: str = None,
                        correct_category: str = None):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
            INSERT INTO feedback (product_id, vote, correct_name, correct_brand, correct_category)
            VALUES (?, ?, ?, ?, ?)
        """, (product_id, vote, correct_name, correct_brand, correct_category))
        # Если дизлайк с исправлением — обновляем продукт
        if vote == -1 and correct_name:
            await db.execute("""
                UPDATE products
                SET product_name=?, brand=?, product_category=?, is_corrected=1
                WHERE id=?
            """, (correct_name, correct_brand or "", correct_category or "Другое", product_id))
        await db.commit()


async def get_feedback_stats():
    """Статистика лайков/дизлайков"""
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute("""
            SELECT
                COUNT(CASE WHEN vote = 1  THEN 1 END) as likes,
                COUNT(CASE WHEN vote = -1 THEN 1 END) as dislikes
            FROM feedback
        """) as cur:
            row = await cur.fetchone()
    return {"likes": row[0], "dislikes": row[1]}

# ══════════════════════════════════════
# GROUP PHOTOS
# ══════════════════════════════════════

async def save_group_photo(file_id: str, file_unique_id: str, chat_id: int):
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
            INSERT OR IGNORE INTO group_photos (file_id, file_unique_id, chat_id)
            VALUES (?, ?, ?)
        """, (file_id, file_unique_id, chat_id))
        await db.commit()


async def get_recent_group_photos(limit: int = 12):
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute("""
            SELECT file_id FROM group_photos
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,)) as cur:
            rows = await cur.fetchall()
    return [row[0] for row in rows]
