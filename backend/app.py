import os
import json
import base64
import datetime
import numpy as np
import pandas as pd
import joblib
from flask import Flask, request, jsonify, send_from_directory
import firebase_admin
from firebase_admin import credentials, db

# --- App Setup ---
app = Flask(__name__, static_folder='../frontend/build/static', template_folder='../frontend/build')

# --- Firebase Init ---
FIREBASE_SERVICE_ACCOUNT_KEY_ENCODED = os.environ.get('FIREBASE_SERVICE_ACCOUNT_KEY')
DATABASE_URL = os.environ.get('FIREBASE_DATABASE_URL')

ref = None

if FIREBASE_SERVICE_ACCOUNT_KEY_ENCODED and DATABASE_URL:
    try:
        decoded_key = base64.b64decode(FIREBASE_SERVICE_ACCOUNT_KEY_ENCODED).decode('utf-8')
        service_account_info = json.loads(decoded_key)
        cred = credentials.Certificate(service_account_info)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {'databaseURL': DATABASE_URL})
        ref = db.reference('structural_data')
        print("✅ Firebase initialized successfully!")
    except Exception as e:
        print(f"❌ Firebase init error: {e}")
else:
    print("⚠️ WARNING: Firebase environment variables not set.")

# --- Load Model ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Model load error: {e}")
    model = None

# --- API Routes ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    try:
        data = request.get_json()
        ax_g = float(data['ax_g'])
        ay_g = float(data['ay_g'])
        az_g = float(data['az_g'])
        vibration = float(data['vibration'])
        bending = float(data['bending'])
    except (KeyError, ValueError) as e:
        return jsonify({'error': str(e)}), 400

    totalAccel = np.sqrt(ax_g**2 + ay_g**2 + az_g**2)
    input_df = pd.DataFrame([{
        'ax_g': ax_g,
        'ay_g': ay_g,
        'az_g': az_g,
        'totalAccel': totalAccel,
        'vibration': vibration,
        'bending': bending
    }])

    try:
        pred = model.predict(input_df)[0]
        status = "DANGER" if pred == 1 else "SAFE"
    except Exception as e:
        return jsonify({'error': f'Prediction error: {e}'}), 500

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

    if ref:
        try:
            ref.push(prediction_data)
        except Exception as e:
            print(f"⚠️ Firebase write failed: {e}")

    return jsonify({'status': status})

# --- Static/Frontend Routes ---
@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    target_path = os.path.join(app.template_folder, path)
    if path != "" and os.path.exists(target_path):
        return send_from_directory(app.template_folder, path)
    else:
        return send_from_directory(app.template_folder, 'index.html')

# --- Entry Point ---
# --- Entrypoint for Render 🚀 ---
if __name__ == '__main__':
    from waitress import serve
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Starting app on 0.0.0.0:{port}")
    serve(app, host='0.0.0.0', port=port)
