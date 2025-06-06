import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify, send_from_directory
import firebase_admin
from firebase_admin import credentials, db
import datetime
import os

app = Flask(__name__, static_folder='../frontend/build/static', template_folder='../frontend/build')

# --- Firebase Initialization ---
SERVICE_ACCOUNT_KEY_PATH = os.path.join(os.path.dirname(__file__), 'serviceAccountKey.json')
DATABASE_URL = 'https://health-3c965-default-rtdb.firebaseio.com'

try:
    cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
    firebase_admin.initialize_app(cred, {
        'databaseURL': DATABASE_URL
    })
    print("Firebase initialized successfully!")
except FileNotFoundError:
    print(f"Error: Service account key not found at {SERVICE_ACCOUNT_KEY_PATH}")
except Exception as e:
    print(f"Firebase init error: {e}")

ref = db.reference('structural_data')

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
try:
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully!")
except FileNotFoundError:
    print(f"Error: model.pkl not found at {MODEL_PATH}")
    model = None
except Exception as e:
    print(f"Model load error: {e}")
    model = None

@app.route('/')
def serve_react_app():
    return render_template('index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded on the server.'}), 500

    data = request.json
    try:
        ax_g = float(data['ax_g'])
        ay_g = float(data['ay_g'])
        az_g = float(data['az_g'])
        vibration = float(data['vibration'])
        bending = float(data['bending'])
    except KeyError as e:
        return jsonify({'error': f'Missing field: {e}'}), 400
    except ValueError as e:
        return jsonify({'error': f'Invalid data: {e}'}), 400

    totalAccel = np.sqrt(ax_g**2 + ay_g**2 + az_g**2)

    input_df = pd.DataFrame([{
        'ax_g': ax_g,
        'ay_g': ay_g,
        'az_g': az_g,
        'totalAccel': totalAccel,
        'vibration': vibration,
        'bending': bending
    }])

    pred = model.predict(input_df)[0]
    status = "DANGER" if pred == 1 else "SAFE"

    timestamp = datetime.datetime.now().isoformat()
    prediction_data = {
        'timestamp': timestamp,
        'ax_g': ax_g,
        'ay_g': ay_g,
        'az_g': az_g,
        'totalAccel': totalAccel,
        'vibration': vibration,
        'bending': bending,
        'predicted_status': status
    }

    try:
        if 'firebase_admin' in globals() and firebase_admin._apps:
            ref.push(prediction_data)
            print(f"Data saved: {prediction_data}")
        else:
            print("Firebase not initialized.")
    except Exception as e:
        print(f"Firebase save error: {e}")

    return jsonify({'status': status})

if __name__ == '__main__':
    # Use waitress for Windows compatibility
    from waitress import serve
    serve(app, host='0.0.0.0', port=5000)
