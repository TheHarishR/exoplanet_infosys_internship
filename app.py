from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load trained ML model
model = pickle.load(open("habitability_model.pkl", "rb"))

@app.route("/")
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predictor")
def predictor():
    return render_template("predictor.html")
@app.route("/dashboard")
def dashboard():
    accuracy = 94
    last_score = 0.78
    predictions = [0.2, 0.35, 0.6, 0.78, 0.82]

    return render_template(
        "dashboard.html",
        accuracy=accuracy,
        last_score=last_score,
        predictions=predictions
    )


@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/ui")
def ui():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)

        # 4 features from UI
        planet_temp_z = float(data["planet_temp_z"])
        planet_size_z = float(data["planet_size_z"])
        star_temp_z = float(data["star_temp_z"])
        star_energy_z = float(data["star_energy_z"])

        # 3 missing features (DEFAULT = 0.0 → mean of z-score)
        star_gravity_z = 0.0
        star_size_z = 0.0
        brightness_z = 0.0

        # TOTAL = 7 FEATURES (MATCH MODEL TRAINING)
        X = np.array([
            planet_temp_z,
            planet_size_z,
            star_temp_z,
            star_energy_z,
            star_gravity_z,
            star_size_z,
            brightness_z
        ]).reshape(1, -1)

        score = model.predict_proba(X)[0][1]
        prediction = "Habitable" if score >= 0.5 else "Non-Habitable"

        return jsonify({
            "habitability_score": round(float(score), 3),
            "prediction": prediction
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
