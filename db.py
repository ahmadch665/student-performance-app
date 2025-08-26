import os
import mysql.connector
from mysql.connector import Error

def get_db_connection():
    """Establish and return a MySQL database connection."""
    try:
        connection = mysql.connector.connect(
            host=os.environ.get('DB_HOST'),
            database=os.environ.get('DB_NAME'),
            user=os.environ.get('DB_USER'),
            password=os.environ.get('DB_PASSWORD')
        )
        return connection
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None


def fetch_users():
    conn = get_db_connection()
    if conn is None:
        return []
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT id, username, email, role_id FROM users")
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return rows


def add_user(username, password_hash, role_id, email):
    conn = get_db_connection()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO users (username, password_hash, role_id, email)
            VALUES (%s, %s, %s, %s)
        """, (username, password_hash, role_id, email))
        conn.commit()
        cursor.close()
        return True
    except Error as e:
        print(f"Error adding user: {e}")
        return False
    finally:
        conn.close()


def update_user_email(username, new_email):
    conn = get_db_connection()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE users SET email=%s WHERE username=%s
        """, (new_email, username))
        conn.commit()
        cursor.close()
        return True
    except Error as e:
        print(f"Error updating email: {e}")
        return False
    finally:
        conn.close()
