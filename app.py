from flask import Flask, render_template, request, redirect, url_for, flash, abort, session, send_file
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from werkzeug.security import generate_password_hash 
import io
import os
import pandas as pd
from utils.preprocessing import preprocess_csv
from models.ml_models import train_and_predict
from fpdf import FPDF

# NEW imports for auth
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bcrypt import Bcrypt
import mysql.connector
from functools import wraps
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from flask import jsonify



app = Flask(__name__)

@app.before_request
def logout_on_root():
    if request.path == '/':
        session.clear()

app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'
# Token serializer for reset links
serializer = URLSafeTimedSerializer(app.secret_key)



bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Ensure uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Fixed required columns
FIXED_COLUMNS = ["student_id", "name", "gender", "age", "semester", "attendance_percent"]

# Database helper
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            database='student_risk_db',
            user='root',
            password='YourNewPassword123!'
        )
        return conn
    except mysql.connector.Error as e:
        print(f"Database connection error: {e}")
        return None

# User class
class User(UserMixin):
    def __init__(self, id, username, role_id):
        self.id = id
        self.username = username
        self.role_id = role_id

@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    if not conn:
        return None
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
    user = cursor.fetchone()
    cursor.close()
    conn.close()
    if user:
        return User(user['id'], user['username'], user['role_id'])
    return None

# Role decorators
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role_id != 1:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

def faculty_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role_id != 2:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

def advisor_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or current_user.role_id != 3:
            abort(403)
        return f(*args, **kwargs)
    return decorated_function

# ---------- Authentication routes ----------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        if current_user.role_id == 1:
            return redirect(url_for('admin_dashboard'))
        elif current_user.role_id == 2:
            return redirect(url_for('faculty_dashboard'))
        elif current_user.role_id == 3:
            return redirect(url_for('advisor_dashboard'))
        else:
            return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = get_db_connection()
        if not conn:
            flash("Database connection error.", "danger")
            return redirect(url_for('login'))

        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if user and bcrypt.check_password_hash(user['password_hash'], password):
            user_obj = User(user['id'], user['username'], user['role_id'])
            login_user(user_obj)
            flash('Logged in successfully!', 'success')
            if user_obj.role_id == 1:
                return redirect(url_for('admin_dashboard'))
            elif user_obj.role_id == 2:
                return redirect(url_for('faculty_dashboard'))
            elif user_obj.role_id == 3:
                return redirect(url_for('advisor_dashboard'))
            else:
                return redirect(url_for('index'))
        else:
            flash('Invalid username or password', 'danger')

    return render_template('login.html')



@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return redirect(url_for('login'))


# Step 1: Ask username
@app.route("/forgot_password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        username = request.form.get("username")  # use .get() to avoid KeyError
        if not username:
            flash("Username is required", "danger")
            return redirect(url_for("forgot_password"))

        # Check if username exists in DB
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()

        if not user:
            flash("No account found with that username.", "danger")
            return redirect(url_for("forgot_password"))

        # If found → ask for email
        return render_template("verify_email.html", username=username)

    return render_template("forgot_password.html")




@app.route("/verify_email", methods=["POST"])
def verify_email():
    username = request.form.get("username")
    email = request.form.get("email")

    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM users WHERE username = %s AND email = %s", (username, email))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if not user:
        flash("Email does not match with username.", "danger")
        return redirect(url_for("forgot_password"))

    # If email matches → show reset form
    return render_template("reset_password.html", username=username)


# Step 3: Reset password
@app.route('/reset_password/<username>', methods=['GET', 'POST'])
def reset_password(username):
    if request.method == 'POST':
        new_password = request.form.get("new_password")
        confirm_password = request.form.get("confirm_password")

        if not new_password or not confirm_password:
            flash("Please fill out all fields", "danger")
            return redirect(request.url)

        if new_password != confirm_password:
            flash("Passwords do not match!", "danger")
            return redirect(request.url)

        # (Optional) simple strength check
        if len(new_password) < 6:
            flash("Password must be at least 6 characters long", "danger")
            return redirect(request.url)

        # IMPORTANT: use Flask-Bcrypt to hash (compatible with your login check)
        hashed_pw = bcrypt.generate_password_hash(new_password).decode('utf-8')

        conn = get_db_connection()
        if not conn:
            flash("Database connection error.", "danger")
            return redirect(request.url)

        cursor = conn.cursor()
        # IMPORTANT: update the correct column name
        cursor.execute("UPDATE users SET password_hash = %s WHERE username = %s", (hashed_pw, username))
        conn.commit()
        cursor.close()
        conn.close()

        flash("Password reset successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template("reset_password.html", username=username)



# ---------- Home / index ----------
@app.route('/')
@login_required
def index():
    return render_template('index.html')

# ---------- Upload route ----------
@app.route('/upload', methods=['POST'])
@login_required
@admin_required
def upload():
    file = request.files.get('dataset')

    if not file or not file.filename.endswith('.csv'):
        flash('❌ Invalid file format. Please upload a .csv file.', 'danger')
        return redirect(url_for('admin_dashboard'))   # back to admin page

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath)

        # Validation rules
        if 'Risk_Level' in df.columns:
            flash("❌ Remove 'Risk_Level' column from CSV.", "danger")
            return redirect(url_for('admin_dashboard'))

        missing_fixed = [col for col in FIXED_COLUMNS if col not in df.columns]
        if missing_fixed:
            flash(f"❌ Missing required columns: {', '.join(missing_fixed)}", "danger")
            return redirect(url_for('admin_dashboard'))

        subject_cols = [col for col in df.columns if col not in FIXED_COLUMNS]
        invalid_subjects = [col for col in subject_cols if not col.endswith('_marks')]
        if invalid_subjects:
            flash(f"❌ Invalid subject column(s): {', '.join(invalid_subjects)}.", "danger")
            return redirect(url_for('admin_dashboard'))

        if not pd.to_numeric(df['semester'], errors='coerce').between(1, 10).all():
            flash("❌ Semester values must be integers between 1 and 10.", "danger")
            return redirect(url_for('admin_dashboard'))

        # If all good → preprocess
        processed_df = preprocess_csv(filepath)
        processed_df.to_csv('uploads/processed_data.csv', index=False)

        flash('✅ File uploaded and preprocessed successfully!', 'success')
        return redirect(url_for('dashboard'))

    except Exception as e:
        print("Preprocessing failed:", e)
        flash(f'Error processing file: {str(e)}', 'danger')
        return redirect(url_for('admin_dashboard'))

# ======= Admin Dashboard Route =======
@app.route('/admin_dashboard')
@login_required
@admin_required
def admin_dashboard():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # fetch faculty + advisors
    cursor.execute("SELECT id, username, email, role_id FROM users WHERE role_id IN (2, 3)")
    faculty_advisors = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('admin_dashboard.html', faculty_advisors=faculty_advisors)

@app.route('/faculty_advisor_list')
@login_required
@admin_required
def faculty_advisor_list():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)

    # Fetch all faculty/advisors with their subjects
    query = """
        SELECT u.id, u.username, u.email, u.role_id,
               GROUP_CONCAT(fs.subject_name SEPARATOR ', ') AS subjects
        FROM users u
        LEFT JOIN faculty_subjects fs ON u.id = fs.user_id
        WHERE u.role_id IN (2, 3)
        GROUP BY u.id, u.username, u.email, u.role_id
    """
    cursor.execute(query)
    faculty_advisors = cursor.fetchall()

    cursor.close()
    conn.close()

    return render_template('faculty_advisor_list.html', faculty_advisors=faculty_advisors)

@app.route('/update_user/<int:user_id>', methods=['POST'])
@login_required
@admin_required
def update_user(user_id):
    username = request.form.get('username')
    email = request.form.get('email')
    role = request.form.get('role')
    subjects_input = request.form.get('subjects')  # comma-separated for faculty

    conn = get_db_connection()
    cursor = conn.cursor()

    # Update user info
    cursor.execute("""
        UPDATE users SET username=%s, email=%s, role_id=%s WHERE id=%s
    """, (username, email, role, user_id))

    # Update faculty subjects
    cursor.execute("DELETE FROM faculty_subjects WHERE user_id=%s", (user_id,))
    if role == "2" and subjects_input:
        for subj in subjects_input.split(','):
            subject_name = subj.strip().lower().replace(" ", "_")
            if not subject_name.endswith("_marks"):
                subject_name += "_marks"
            cursor.execute(
                "INSERT INTO faculty_subjects (user_id, subject_name) VALUES (%s, %s)",
                (user_id, subject_name)
            )

    conn.commit()
    cursor.close()
    conn.close()

    flash("User updated successfully!", "success")
    return redirect(url_for('faculty_advisor_list'))


@app.route('/delete_user/<int:user_id>')
@login_required
@admin_required
def delete_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Delete user subjects first (if any)
    cursor.execute("DELETE FROM faculty_subjects WHERE user_id=%s", (user_id,))
    
    # Delete user
    cursor.execute("DELETE FROM users WHERE id=%s", (user_id,))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    flash("User deleted successfully!", "success")
    return redirect(url_for('faculty_advisor_list'))

# ======= Run Model Route =======
@app.route('/run_model')
@login_required
@admin_required
def run_model():
    try:
        filepath = 'uploads/processed_data.csv'
        df, best_model, accuracy = train_and_predict(filepath)

        if 'Prediction' in df.columns:
            df.drop(columns=['Prediction'], inplace=True)

        df.to_csv('uploads/final_predictions.csv', index=False)
        flash(f'Model run successful! Accuracy: {round(accuracy * 100, 2)}%', 'success')
        return redirect(url_for('dashboard'))

    except Exception as e:
        print("Model run failed:", e)
        flash(f'Model run failed: {str(e)}', 'danger')
        return redirect(url_for('admin_dashboard'))



# ======= Dashboard Route =======
# ======= Dashboard Route =======
@app.route('/dashboard')
@login_required
def dashboard():
    # Define default suggested model for first load
    default_best_model = 'DecisionTree'  # Can be any model you want to suggest initially

    return render_template('dashboard.html',
                           model=None,
                           accuracy=None,
                           table=None,
                           students=[],
                           columns=[],
                           risk_data={},
                           subject_data={},
                           trend_data={},
                           semesters=[],
                           student_radar_all={},
                           student_names=[],
                           best_model_name=default_best_model)  # Pass suggested model


# ======= Predict Route =======
@app.route('/predict')
@login_required
def predict():
    try:
        # Get selected model from dropdown, default to SVM
        selected_model = request.form.get("model")


        filepath = 'uploads/processed_data.csv'
        # Updated: capture all model results
        df, best_model_name, best_accuracy, all_results = train_and_predict(filepath, selected_model)

        if 'Prediction' in df.columns:
            df.drop(columns=['Prediction'], inplace=True)

        # Prepare table
        table_html = df.to_html(classes='table table-striped student-data-table', index=False)
        students = df.to_dict(orient='records')
        columns = df.columns.tolist()

        # Pie chart: Risk distribution
        risk_counts = df['Risk_Level'].value_counts().to_dict()

        # Bar chart: Subject averages
        subject_averages = {}
        subject_columns = [col for col in df.columns if col.endswith('_marks')]
        for col in subject_columns:
            pretty_label = col.replace('_marks', '').capitalize()
            subject_averages[pretty_label] = round(df[col].mean(), 2)

        # Trend chart
        trend_data = {}
        semesters = []
        if 'semester' in df.columns:
            df['semester'] = pd.to_numeric(df['semester'], errors='coerce')
            df.dropna(subset=['semester'], inplace=True)
            df['semester'] = df['semester'].astype(int)
            semesters = sorted(df['semester'].unique().tolist())
            for col in subject_columns:
                pretty_label = col.replace('_marks', '').capitalize()
                semester_means = df.groupby('semester')[col].mean().reindex(semesters).fillna(0).round(2).tolist()
                trend_data[pretty_label] = semester_means

        # Radar chart
        student_names = df['name'].unique().tolist()
        student_radar_all = {}
        for name in student_names:
            student_row = df[df['name'] == name].iloc[0]
            student_radar_all[name] = {}
            for col in subject_columns:
                pretty_label = col.replace('_marks', '').capitalize()
                student_radar_all[name][pretty_label] = float(student_row[col])

        # Weak/Focus areas
        student_focus_areas = {}
        for name in student_names:
            row = df[df['name'] == name].iloc[0]
            focus_list = []
            for col in subject_columns:
                if float(row[col]) < 50:
                    focus_list.append(col.replace('_marks', '').capitalize())
            if float(row['attendance_percent']) < 75:
                focus_list.append('Attendance')
            student_focus_areas[name] = focus_list

        return render_template('dashboard.html',
                               model=best_model_name,
                               accuracy=round(best_accuracy * 100, 2),
                               table=table_html,
                               students=students,
                               columns=columns,
                               risk_data=risk_counts,
                               subject_data=subject_averages,
                               trend_data=trend_data,
                               semesters=semesters,
                               student_radar_all=student_radar_all,
                               student_names=student_names,
                               student_focus_areas=student_focus_areas,
                               all_model_results=all_results)  # Pass all model accuracies

    except Exception as e:
        print("Prediction failed:", e)
        flash(f"Prediction failed: {str(e)}", 'danger')
        return redirect(url_for('dashboard'))



# ======= Faculty Dashboard Route =======
@app.route('/faculty_dashboard')
@login_required
@faculty_required
def faculty_dashboard():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT subject_name FROM faculty_subjects WHERE user_id = %s", (current_user.id,))
    subjects = cursor.fetchall()
    cursor.close()
    conn.close()

    if not subjects:
        flash("No subjects assigned to you yet.", "warning")
        return render_template('faculty_dashboard.html', students=[], columns=[], subject_averages={}, risk_data={},
                               trend_data={}, semesters=[], student_radar_all={}, student_names=[])

    subject_cols = [s['subject_name'] for s in subjects]

    try:
        df = pd.read_csv('uploads/final_predictions.csv')
    except FileNotFoundError:
        flash("No prediction data available. Please contact admin.", "danger")
        return render_template('faculty_dashboard.html', students=[], columns=[], subject_averages={}, risk_data={},
                               trend_data={}, semesters=[], student_radar_all={}, student_names=[])

    fixed_cols = ["student_id", "name", "gender", "age", "semester", "attendance_percent"]
    needed_cols = fixed_cols + subject_cols + ['Risk_Level']
    df_filtered = df.loc[:, [col for col in needed_cols if col in df.columns]]

    students = df_filtered.to_dict(orient='records')
    columns = df_filtered.columns.tolist()

    risk_counts = df_filtered['Risk_Level'].value_counts().to_dict() if 'Risk_Level' in df_filtered.columns else {}

    subject_averages = {}
    for col in subject_cols:
        if col in df_filtered.columns:
            pretty_label = col.replace('_marks', '').capitalize()
            subject_averages[pretty_label] = round(df_filtered[col].mean(), 2)

    trend_data = {}
    semesters = []
    if 'semester' in df_filtered.columns:
        df_filtered['semester'] = pd.to_numeric(df_filtered['semester'], errors='coerce')
        df_filtered.dropna(subset=['semester'], inplace=True)
        df_filtered['semester'] = df_filtered['semester'].astype(int)
        semesters = sorted(df_filtered['semester'].unique().tolist())
        for col in subject_cols:
            pretty_label = col.replace('_marks', '').capitalize()
            semester_means = df_filtered.groupby('semester')[col].mean().reindex(semesters).fillna(0).round(2).tolist()
            trend_data[pretty_label] = semester_means

    student_names = df_filtered['name'].unique().tolist()
    student_radar_all = {}
    for name in student_names:
        student_row = df_filtered[df_filtered['name'] == name].iloc[0]
        student_radar_all[name] = {}
        for col in subject_cols:
            pretty_label = col.replace('_marks', '').capitalize()
            student_radar_all[name][pretty_label] = float(student_row[col])

    return render_template('faculty_dashboard.html',
                           students=students,
                           columns=columns,
                           subject_averages=subject_averages,
                           risk_data=risk_counts,
                           trend_data=trend_data,
                           semesters=semesters,
                           student_radar_all=student_radar_all,
                           student_names=student_names)




# ======= Advisor Dashboard Route =======
@app.route('/advisor_dashboard')
@login_required
@advisor_required
def advisor_dashboard():
    try:
        df = pd.read_csv('uploads/final_predictions.csv')
    except FileNotFoundError:
        flash("No prediction data available. Please contact admin.", "danger")
        return render_template(
            'advisor_dashboard.html',
            students=[], columns=[], subject_columns=[],
            risk_data={}, attention_students=[]
        )

    # === Advisor sees ALL students ===
    students = df.to_dict(orient='records')
    columns = df.columns.tolist()

    # --- Subject columns for comparison ---
    # Assuming all columns ending with '_marks' are subjects
    subject_columns = [col for col in df.columns if col.endswith('_marks')]

    # --- Risk pie data ---
    risk_counts = df['Risk_Level'].value_counts().to_dict() if 'Risk_Level' in df.columns else {}

    # --- Students Requiring Attention (all Risk students) ---
    if 'Risk_Level' in df.columns:
        attention_df = df[df['Risk_Level'].str.lower() == 'risk'].copy()
    else:
        attention_df = df.iloc[0:0]  # empty

    # Select only the columns we display in the table
    show_cols = ['student_id', 'name', 'attendance_percent', 'Risk_Level']
    show_cols = [c for c in show_cols if c in attention_df.columns]
    attention_students = attention_df[show_cols].to_dict(orient='records')

    return render_template(
        'advisor_dashboard.html',
        students=students,
        columns=columns,
        subject_columns=subject_columns,
        risk_data=risk_counts,
        attention_students=attention_students
    )






@app.route('/student_profile/<student_id>')
@login_required
@advisor_required
def student_profile(student_id):
    try:
        df = pd.read_csv('uploads/final_predictions.csv')
    except FileNotFoundError:
        flash("No prediction data available. Please contact admin.", "danger")
        return redirect(url_for('advisor_dashboard'))

    # Filter student by ID
    student_df = df[df['student_id'] == int(student_id)]
    if student_df.empty:
        flash("Student not found.", "danger")
        return redirect(url_for('advisor_dashboard'))

    student = student_df.iloc[0].to_dict()

    # Subjects (any column ending with '_marks')
    subject_cols = [col for col in df.columns if col.endswith('_marks')]

    # Semester-wise marks trend (if you have semester column)
    if 'semester' in df.columns:
        trend_df = df[df['student_id'] == int(student_id)].sort_values('semester')
        semesters = trend_df['semester'].tolist()
        trend_marks = {subj: trend_df[subj].tolist() for subj in subject_cols}
    else:
        semesters = ['1']
        trend_marks = {subj: [student[subj]] for subj in subject_cols}

    # Radar chart (subject-wise performance)
    radar_data = {subj: student[subj] for subj in subject_cols}

    # Attendance
    attendance = student.get('attendance_percent', 0)

    # Past Risk Levels
    if 'Risk_Level' in df.columns:
        past_risk = df[df['student_id'] == int(student_id)].sort_values('semester')['Risk_Level'].tolist()
    else:
        past_risk = [student.get('Risk_Level', '')]

    return render_template(
        'student_profile.html',
        student=student,
        semesters=semesters,
        trend_marks=trend_marks,
        radar_data=radar_data,
        attendance=attendance,
        past_risk=past_risk
    )


# ======= Add User Route =======
@app.route('/add_user', methods=['GET', 'POST'])
@login_required
@admin_required
def add_user():
    if request.method == 'POST':
        try:
            username = request.form.get('username')
            password = request.form.get('password')
            role = request.form.get('role')   # 2 for Faculty, 3 for Advisor
            email = request.form.get('email')
            subject_input = request.form.get('subject')  # optional for faculty

            if not username or not password or not role or not email:
                flash("All fields are required.", "danger")
                return redirect(url_for('add_user'))

            # Hash password using Flask-Bcrypt
            password_hash = bcrypt.generate_password_hash(password).decode('utf-8')

            # Insert user into database
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (username, password_hash, role_id, email)
                VALUES (%s, %s, %s, %s)
            """, (username, password_hash, role, email))

            # Get the newly inserted user's ID
            new_user_id = cursor.lastrowid

            # If role is faculty (2) and subject is provided, insert subjects
            if role == "2" and subject_input:
                subjects = [s.strip().lower().replace(" ", "_") + "_marks" for s in subject_input.split(',')]
                for subject_name in subjects:
                    cursor.execute("""
                        INSERT INTO faculty_subjects (user_id, subject_name)
                        VALUES (%s, %s)
                    """, (new_user_id, subject_name))

            # Commit everything at once
            conn.commit()
            cursor.close()
            conn.close()

            flash(f"User '{username}' added successfully!", "success")
            return redirect(url_for('add_user'))

        except Exception as e:
            flash(f"Failed to add user: {str(e)}", "danger")
            return redirect(url_for('add_user'))

    # GET request: show add_user page
    return render_template('add_user.html')




# ======= Export CSV Route =======
@app.route('/export_csv')
@login_required
def export_csv():
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'final_predictions.csv')
        if not os.path.exists(file_path):
            flash("No prediction file found. Please run the model first.", "danger")
            return redirect(url_for('faculty_dashboard'))

        df = pd.read_csv(file_path)

        # If current user is faculty, filter columns to the same set shown on the faculty page
        if current_user.is_authenticated and getattr(current_user, 'role_id', None) == 2:
            conn = get_db_connection()
            if not conn:
                flash("Database connection error while preparing export.", "danger")
                return redirect(url_for('faculty_dashboard'))
            try:
                cursor = conn.cursor(dictionary=True)
                cursor.execute("SELECT subject_name FROM faculty_subjects WHERE user_id = %s", (current_user.id,))
                subjects_rows = cursor.fetchall()
            finally:
                try:
                    cursor.close()
                except:
                    pass
                conn.close()

            subject_cols = [r['subject_name'] for r in subjects_rows] if subjects_rows else []
            fixed_cols = ["student_id", "name", "gender", "age", "semester", "attendance_percent"]
            needed_cols = fixed_cols + subject_cols + ['Risk_Level']
            columns_to_keep = [c for c in needed_cols if c in df.columns]

            if not columns_to_keep:
                flash("No matching columns found to export for your account.", "warning")
                return redirect(url_for('faculty_dashboard'))

            export_df = df.loc[:, columns_to_keep]
        else:
            export_df = df

        output = io.BytesIO()
        csv_bytes = export_df.to_csv(index=False).encode('utf-8')
        output.write(csv_bytes)
        output.seek(0)

        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='prediction_results.csv'
        )

    except Exception as e:
        flash(f"Failed to export CSV: {str(e)}", "danger")
        return redirect(url_for('faculty_dashboard'))

@app.route('/add_user_page')
@login_required
@admin_required
def add_user_page():
    return render_template('add_user.html')




# ======= App Run ======#
if __name__ == '__main__':
    app.run(debug=True)
