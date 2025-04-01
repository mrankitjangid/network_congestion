from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scalers
model = load_model("lstm_model.h5", compile=False)
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = ['bandwidth', 'throughput', 'packet_loss', 'latency', 'jitter']
        input_data = np.array([data[feature] for feature in features]).reshape(1, -1)
        
        # Scale input
        input_scaled = scaler_X.transform(input_data)
        input_reshaped = np.expand_dims(input_scaled, axis=1)  # Reshape for LSTM
        
        # Predict and inverse transform
        prediction_scaled = model.predict(input_reshaped)
        prediction = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0][0]
        
        return jsonify({"predicted_congestion": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
