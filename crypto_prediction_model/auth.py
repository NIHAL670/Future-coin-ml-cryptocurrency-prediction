import sqlite3
import bcrypt

# ---------------- DATABASE ----------------
def get_db():
    return sqlite3.connect("users.db", check_same_thread=False)

def create_users_table():
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            email TEXT UNIQUE,
            password BLOB
        )
    """)
    conn.commit()
    conn.close()

# ---------------- SIGN UP ----------------
def register_user(username, email, password):
    conn = get_db()
    c = conn.cursor()

    hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

    try:
        c.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username, email, hashed_pw)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

# ---------------- LOGIN ----------------
def authenticate_user(username_or_email, password):
    conn = get_db()
    c = conn.cursor()

    c.execute(
        "SELECT password FROM users WHERE username=? OR email=?",
        (username_or_email, username_or_email)
    )
    result = c.fetchone()
    conn.close()

    if result and bcrypt.checkpw(password.encode(), result[0]):
        return True
    return False
