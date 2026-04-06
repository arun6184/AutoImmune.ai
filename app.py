from flask import Flask, render_template, request, redirect, session, url_for, g
import pickle
import numpy as np
import joblib
import os
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# --- Database Setup ---
DATABASE = 'users.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def query_db(query, args=(), one=False):
    cur = get_db().execute(query, args)
    rv = cur.fetchall()
    cur.close()
    return (rv[0] if rv else None) if one else rv

def init_db():
    db = get_db()
    db.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    db.commit()

# --- Load Models ---
with open('model.pickle', 'rb') as file:
    model = pickle.load(file)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

flare_model = joblib.load("flare_predictor.pkl")


# --- User Auth Routes ---

def initialize():
    init_db()

@app.route('/')
def index():
    if 'username' in session:
        return redirect(url_for('home'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    message = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if query_db('SELECT * FROM users WHERE username = ?', [username], one=True):
            message = "Username already exists!"
        else:
            hashed_password = generate_password_hash(password)
            db = get_db()
            db.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
            db.commit()
            return redirect(url_for('login'))
    return render_template('register.html', message=message)

@app.route('/login', methods=['GET', 'POST'])
def login():
    message = ''
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = query_db('SELECT * FROM users WHERE username = ?', [username], one=True)
        if user and check_password_hash(user[2], password):
            session['username'] = username
            return redirect(url_for('home'))
        else:
            message = 'Invalid username or password!'
    return render_template('login.html', message=message)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

# --- Main Application Pages ---

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', username=session['username'])


@app.route('/clinical')
def clinical_page():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('clinical_predict.html')

@app.route('/predict_clinical', methods=['POST'])
def predict_clinical():
    if 'username' not in session:
        return redirect(url_for('login'))

    try:
        gender_map = {"Male": 0, "Female": 1}
        diagnosis_labels = {
            0: 'Autoimmune orchitis',
            1: "Graves' disease",
            2: 'Rheumatoid arthritis',
            3: 'Sjögren syndrome',
            4: 'Systemic lupus erythematosus (SLE)'
        }
        recommendations = {
            'Autoimmune orchitis': "Get plenty of rest, take prescribed anti-inflammatory medications, and avoid physical strain.",
            "Graves' disease": "Follow thyroid treatment regularly, manage stress, and avoid excessive iodine.",
            'Rheumatoid arthritis': "Stay active with low-impact exercise, use joint supports, and follow a balanced diet.",
            'Sjögren syndrome': "Use eye drops, stay hydrated, and avoid dry or dusty environments.",
            'Systemic lupus erythematosus (SLE)': "Avoid sun exposure, take medications on time, and get regular checkups."
        }
        data = [
            request.form['id'],
            request.form['age'],
            gender_map[request.form['gender']],
            request.form['sickness_duration'],
            request.form['rbc_count'],
            request.form['hemoglobin'],
            request.form['hematocrit'],
            request.form['mcv'],
            request.form['mch'],
            request.form['mchc'],
            request.form['rdw'],
            request.form['reticulocyte_count'],
            request.form['wbc_count'],
            request.form['neutrophils'],
            request.form['lymphocytes'],
            request.form['monocytes'],
            request.form['eosinophils'],
            request.form['basophils'],
            request.form['plt_count'],
            request.form['mpv'],
            request.form['ana'],
            request.form['esbach'],
            request.form['mbl_level'],
            request.form['esr'],
            request.form['c3'],
            request.form['c4'],
            request.form['crp']
        ]

        features = np.array(data, dtype=float).reshape(1, -1)
        prediction = model.predict(features)[0]
        predicted_diagnosis = diagnosis_labels.get(prediction, "Unknown Diagnosis")
        suggestion = recommendations.get(predicted_diagnosis, "Consult a specialist for proper care.")
        return render_template('clinical_predict.html',
                               prediction=f"Predicted Diagnosis: {predicted_diagnosis}",
                               suggestion=suggestion)

    except Exception as e:
        return render_template('clinical_predict.html', prediction=f"Error: {str(e)}")


@app.route('/symptom_predict', methods=['GET', 'POST'])
def symptom_predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        file = request.files['note_file']
        if file and file.filename.endswith('.txt'):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            with open(filepath, 'r') as f:
                note = f.read()

            pred = flare_model.predict([note])[0]
            result = "Flare" if pred == 1 else "No Flare"
            suggestions = {
                "Flare": "Get plenty of rest, take medications as prescribed, avoid stress, and stay hydrated. Consult your doctor for further treatment.",
                "No Flare": "Continue regular treatment and monitoring. Maintain a healthy lifestyle to prevent future flares."
            }

            return render_template('symptom_predict.html',
                                   note=note,
                                   prediction=result,
                                   suggestion=suggestions.get(result))
        else:
            return render_template('symptom_predict.html', error="Please upload a valid .txt file.")

    return render_template('symptom_predict.html')


if __name__ == '__main__':
    app.run(debug=True)
