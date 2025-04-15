from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from keras.models import load_model
from keras.losses import mean_squared_error as mse  # Add this import
import pickle

app = Flask(__name__)
CORS(app)

# Load model and encoders
model = load_model("./model/ipl_mlp_model.h5")
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
        print(label_encoders["bat_team"].classes_)
        print(label_encoders["bowl_team"].classes_)
        batting_team = label_encoders["bat_team"].transform([data["batting_team"]])[0]
        bowling_team = label_encoders["bowl_team"].transform([data["bowling_team"]])[0]

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