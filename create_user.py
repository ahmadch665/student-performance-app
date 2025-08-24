import mysql.connector
from flask_bcrypt import Bcrypt
from flask import Flask

app = Flask(__name__)
bcrypt = Bcrypt(app)

def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        database='student_risk_db',
        user='root',
        password='YourNewPassword123!'  # <-- put your MySQL password here
    )

def create_user(username, password, role_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    password_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    cursor.execute(
        "INSERT INTO users (username, password_hash, role_id) VALUES (%s, %s, %s)",
        (username, password_hash, role_id)
    )
    conn.commit()
    cursor.close()
    conn.close()
    print(f"User '{username}' created successfully with role_id {role_id}.")

if __name__ == "__main__":
    # Example: create users here
    create_user('admin', 'adminpassword', 1)  # Admin role_id = 1
    create_user('faculty1', 'facultypass', 2)  # Faculty role_id = 2
    create_user('advisor1', 'advisorpass', 3)  # Academic Advisor role_id = 3
