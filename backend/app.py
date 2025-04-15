from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from keras.models import load_model
import tensorflow as tf
from keras import losses
import pickle

app = Flask(__name__)
CORS(app)

# Register the 'mse' loss function explicitly
# This covers both Keras 2.x and TF 2.x approaches
losses.mse = tf.keras.losses.MeanSquaredError()
tf.keras.losses.mse = tf.keras.losses.MeanSquaredError()
tf.keras.metrics.mse = tf.keras.metrics.MeanSquaredError()

# Define custom_objects dictionary with all possible MSE references
custom_objects = {
    'mse': tf.keras.losses.MeanSquaredError(),
    'mean_squared_error': tf.keras.losses.MeanSquaredError()
}

# Load model with custom objects
model = load_model("./model/ipl_mlp_model.h5", custom_objects=custom_objects)

with open("./model/label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

# Load scaler
with open("./model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    print("Received data:", data)  # Debugging line
    print("Type of label_encoders:", type(label_encoders))
    print("Keys in label_encoders:", label_encoders.keys())

    try:
        # Encode categorical inputs
        print(label_encoders["batting_team"].classes_)
        print(label_encoders["bowling_team"].classes_)
        batting_team = label_encoders["batting_team"].transform([data["batting_team"]])[0]
        bowling_team = label_encoders["bowling_team"].transform([data["bowling_team"]])[0]

        # Create features array and scale it
        features = np.array([[batting_team, bowling_team, data["overs"], data["runs"], data["wickets"]]])
        features_scaled = scaler.transform(features)  # Make sure to use the scaled features
        
        prediction = model.predict(features_scaled)
        print("Prediction:", prediction)  # Debugging line
        print("Prediction shape:", prediction.shape)  # Debugging line

        return jsonify({"predicted_total": float(prediction[0][0])})

    except Exception as e:
        print("Error:", e)  # More logging
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)