import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

app = Flask(__name__)

# Load the trained model and scalers
model = load_model("lstm_model.h5")
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Define features (must match training)
features = ['bandwidth', 'throughput', 'packet_loss', 'latency', 'jitter', 'latency_ma', 'jitter_ma']

@app.route('/')
def home():
    return "Congestion Prediction API is running. Use /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        
        # Feature Engineering: Moving averages
        df['latency_ma'] = df['latency'].rolling(window=3, min_periods=1).mean()
        df['jitter_ma'] = df['jitter'].rolling(window=3, min_periods=1).mean()
        
        # Normalize input
        X_scaled = scaler_X.transform(df[features])
        
        # Ensure input shape is correct (batch_size=1, sequence_length, num_features)
        X_reshaped = np.expand_dims(X_scaled, axis=0)
        
        # Make prediction
        y_pred = model.predict(X_reshaped)
        y_pred_original = scaler_y.inverse_transform(y_pred)
        
        return jsonify({"predicted_congestion": float(y_pred_original[0][0])})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
