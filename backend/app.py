import os
import json
import base64
import datetime
import numpy as np
import pandas as pd
import joblib
from flask import Flask, send_from_directory, request, jsonify
import firebase_admin
from firebase_admin import credentials, db

# === Flask Setup ===
app = Flask(__name__, static_folder='../frontend/build/static', template_folder='../frontend/build')


# === Firebase Setup ===
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
    print("⚠️ Firebase config not found.")

# === Load Model ===
try:
    model = joblib.load('model.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    model = None
    print(f"❌ Model load error: {e}")

# === Serve React Frontend ===
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react_app(path):
    build_path = os.path.join(app.template_folder, path)
    if path != "" and os.path.exists(build_path):
        return send_from_directory(app.template_folder, path)
    else:
        return send_from_directory(app.template_folder, 'index.html')

# === Prediction Endpoint ===
@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500

    try:
        data = request.get_json()
        ax_g = float(data['ax_g'])
        ay_g = float(data['ay_g'])
        az_g = float(data['az_g'])
        vibration = float(data['vibration'])
        bending = float(data['bending'])

        totalAccel = np.sqrt(ax_g**2 + ay_g**2 + az_g**2)
        input_df = pd.DataFrame([{
            'ax_g': ax_g,
            'ay_g': ay_g,
            'az_g': az_g,
            'totalAccel': totalAccel,
            'vibration': vibration,
            'bending': bending
        }])

        prediction = model.predict(input_df)[0]
        status = "DANGER" if prediction == 1 else "SAFE"

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
            ref.push(prediction_data)
            print("✅ Saved to Firebase:", prediction_data)
        else:
            print("⚠️ Firebase not available")

        return jsonify({'status': status})
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500

# === Production Server Entry Point ===
if __name__ == '__main__':
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
