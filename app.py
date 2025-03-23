from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os

# Initialize Flask app
app = Flask(__name__)

# Load models
model_path = "/Users/parasmotwani/Desktop/PROJECTS/cropYield/models"
best_xgboost = pickle.load(open(os.path.join(model_path, "best_xgboost.pkl"), "rb"))
preprocessor = pickle.load(open(os.path.join(model_path, "preprocessor.pkl"), "rb"))

# Load dataset for dropdown options
df = pd.read_csv("/Users/parasmotwani/Desktop/PROJECTS/cropYield/yield_df.csv")

# Extract unique values for dropdowns
countries = sorted(df["Area"].dropna().unique().tolist())
crops = sorted(df["Item"].dropna().unique().tolist())

@app.route("/")
def index():
    return render_template("index.html", countries=countries, crops=crops)

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            # Convert numeric inputs properly
            Year = int(request.form["Year"])
            average_rain_fall_mm_per_year = float(request.form["average_rain_fall_mm_per_year"])
            pesticides_tonnes = float(request.form["pesticides_tonnes"])
            avg_temp = float(request.form["avg_temp"])
            Area = request.form["Area"]
            Item = request.form["Item"]

            # Ensure categorical values are in the correct format
            features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)

            # Transform features
            transformed_features = preprocessor.transform(features)

            prediction = best_xgboost.predict(transformed_features)[0] 

            return render_template("index.html", prediction=round(prediction, 2), countries=countries, crops=crops)
        except Exception as e:
            return render_template("index.html", prediction=f"Error: {str(e)}", countries=countries, crops=crops)

if __name__ == "__main__":
    app.run(debug=True)
