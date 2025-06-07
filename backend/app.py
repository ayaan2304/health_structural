import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, send_from_directory
import firebase_admin
from firebase_admin import credentials, db
import datetime
import os
import json

# --- Flask App Initialization ---
app = Flask(__name__, static_folder='../frontend/build/static', template_folder='../frontend/build')

# --- Firebase Initialization (Plain JSON file) ---
FIREBASE_JSON_PATH = os.path.join(os.path.dirname(__file__), 'firebase-key.json')
DATABASE_URL = os.environ.get('FIREBASE_DATABASE_URL', 'https://health-3c965-default-rtdb.firebaseio.com')

ref = None  # Firebase DB reference

if os.path.exists(FIREBASE_JSON_PATH):
    try:
        cred = credentials.Certificate(FIREBASE_JSON_PATH)

        # Avoid initializing Firebase more than once
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})

        print("✅ Firebase initialized successfully!")
        ref = db.reference('structural_data')
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not initialize Firebase: {e}")
else:
    print(f"⚠️ WARNING: Firebase credential file not found at {FIREBASE_JSON_PATH}")

# --- Load Model ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except FileNotFoundError:
    print(f"❌ Error: model.pkl not found at {MODEL_PATH}")
    model = None
except Exception as e:
    print(f"❌ Model load error: {e}")
    model = None

# --- Routes ---
@app.route('/')
def serve_react_app():
    return send_from_directory(os.path.join(app.root_path, '../frontend/build'), 'index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on the server.'}), 500

    data = request.get_json()

    # Validate input
    try:
        ax_g = float(data['ax_g'])
        ay_g = float(data['ay_g'])
        az_g = float(data['az_g'])
        vibration = float(data['vibration'])
        bending = float(data['bending'])
    except KeyError as e:
        return jsonify({'error': f'Missing field: {e}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid value: {e}'}), 400

    # Compute features
    totalAccel = np.sqrt(ax_g**2 + ay_g**2 + az_g**2)

    input_df = pd.DataFrame([{
        'ax_g': ax_g,
        'ay_g': ay_g,
        'az_g': az_g,
        'totalAccel': totalAccel,
        'vibration': vibration,
        'bending': bending
    }])

    # Make prediction
    try:
        pred = model.predict(input_df)[0]
        status = "DANGER" if pred == 1 else "SAFE"
    except Exception as e:
        return jsonify({'error': f'Prediction error: {e}'}), 500

    # Prepare data for Firebase
    prediction_data = {
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'ax_g': ax_g,
        'ay_g': ay_g,
        'az_g': az_g,
        'totalAccel': totalAccel,
        'vibration': vibration,
        'bending': bending,
        'predicted_status': status
    }

    # Save to Firebase
    try:
        if ref:
            ref.push(prediction_data)
            print(f"✅ Data saved to Firebase: {prediction_data}")
        else:
            print("⚠️ Firebase reference not available. Data not saved.")
    except Exception as e:
        print(f"❌ Firebase save error: {e}")

    return jsonify({'status': status})

# --- Run App ---
if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
