import aiosqlite

DB_FILE = "market_visit.db"


async def init_db():
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
        # Фото из группы — храним только file_id, файлы не скачиваем
        await db.execute("""
            CREATE TABLE IF NOT EXISTS group_photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_id TEXT NOT NULL,
                file_unique_id TEXT UNIQUE NOT NULL,
                chat_id INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
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
# GROUP PHOTOS
# ══════════════════════════════════════

async def save_group_photo(file_id: str, file_unique_id: str, chat_id: int):
    """Сохраняет file_id фото из группы. Дубликаты игнорируются."""
    async with aiosqlite.connect(DB_FILE) as db:
        await db.execute("""
            INSERT OR IGNORE INTO group_photos (file_id, file_unique_id, chat_id)
            VALUES (?, ?, ?)
        """, (file_id, file_unique_id, chat_id))
        await db.commit()


async def get_recent_group_photos(limit: int = 12):
    """Возвращает последние file_id фото из группы."""
    async with aiosqlite.connect(DB_FILE) as db:
        async with db.execute("""
            SELECT file_id FROM group_photos
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,)) as cur:
            rows = await cur.fetchall()
    return [row[0] for row in rows]