from flask import Flask, request, jsonify
import pickle
from datetime import datetime, date
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --------------------------------
# LOAD LOCAL MODEL
# --------------------------------
MODEL_PATH = "model.pkl"

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)
    model = None


# --------------------------------
# LABEL ENCODING DICTS
# --------------------------------
airline_dict = {
    'AirAsia': 0,
    "Indigo": 1,
    "GO_FIRST": 2,
    "SpiceJet": 3,
    "Air_India": 4,
    "Vistara": 5
}

source_dict = {
    'Delhi': 0,
    "Hyderabad": 1,
    "Bangalore": 2,
    "Mumbai": 3,
    "Kolkata": 4,
    "Chennai": 5
}

departure_dict = {
    'Early_Morning': 0,
    "Morning": 1,
    "Afternoon": 2,
    "Evening": 3,
    "Night": 4,
    "Late_Night": 5
}

stops_dict = {
    'zero': 0,
    "one": 1,
    "two_or_more": 2
}

arrival_dict = {
    'Early_Morning': 0,
    "Morning": 1,
    "Afternoon": 2,
    "Evening": 3,
    "Night": 4,
    "Late_Night": 5
}

destination_dict = {
    'Delhi': 0,
    "Hyderabad": 1,
    "Mumbai": 2,
    "Bangalore": 3,
    "Chennai": 4,
    "Kolkata": 5
}

class_dict = {
    'Economy': 0,
    'Business': 1
}


# --------------------------------
# PREDICTION ENDPOINT
# --------------------------------
@app.route('/predict', methods=['POST'])
def predict():

    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json

        # Convert input into model features
        features = [
            airline_dict[data['airline']],
            source_dict[data['source_city']],
            departure_dict[data['departure_time']],
            stops_dict[data['stops']],
            arrival_dict[data['arrival_time']],
            destination_dict[data['destination_city']],
            class_dict[data['class']],
            max(
                (datetime.strptime(data['departure_date'], '%Y-%m-%d').date() - date.today()).days,
                0
            )
        ]

        prediction = model.predict([features])[0]

        return jsonify({
            "prediction": round(float(prediction), 2)
        })

    except KeyError as e:
        return jsonify({
            "error": f"Missing field: {str(e)}"
        }), 400

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------
# HEALTH CHECK
# --------------------------------
@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None
    })


# --------------------------------
# RUN SERVER
# --------------------------------
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
