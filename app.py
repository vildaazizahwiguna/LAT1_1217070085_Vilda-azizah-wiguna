from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the model outside the route functions for better performance
with open("D:\OneDrive\Dokumen\tutorialFlask\model\hasil_pelatihan_model.pkl", "rb") as mul_reg:
    ml_model = joblib.load(mul_reg)


@app.route("/")
def home():
    """Renders the home page template."""
    return render_template("D:\OneDrive\Dokumen\tutorialFlask\template\home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """Handles prediction requests."""
    print("Prediksi dimulai")
    if request.method == "POST":
        try:
            # Extract form data and convert to floats
            rnd_spend = float(request.form["RnD_Spend"])
            admin_spend = float(request.form["Admin_Spend"])
            market_spend = float(request.form["Market_Spend"])

            # Prepare prediction arguments
            pred_args = [rnd_spend, admin_spend, market_spend]
            pred_args_arr = np.array(pred_args).reshape(1, -1)

            # Make prediction
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction), 2)

            # Render predict template with prediction data
            return render_template("D:\OneDrive\Dokumen\tutorialFlask\template\predict.html", prediction=model_prediction)
        except ValueError:
            return "Please check if the values are entered correctly"

    # Render predict template for GET requests (initial page load)
    return render_template("D:\OneDrive\Dokumen\tutorialFlask\template\predict.html", prediction=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0")