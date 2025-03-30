from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

# Initialize Flask app
app = Flask(__name__)

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current directory
MODEL_DIR = os.path.join(BASE_DIR, "models")  # Define models directory

model_path = os.path.join(MODEL_DIR, "best_xgboost.pkl")
preprocessor_path = os.path.join(MODEL_DIR, "preprocessor.pkl")

try:
    with open(model_path, "rb") as model_file:
        best_xgboost = pickle.load(model_file)

    with open(preprocessor_path, "rb") as preprocessor_file:
        preprocessor = pickle.load(preprocessor_file)
except Exception as e:
    print(f"Error loading models: {e}")
    best_xgboost = None
    preprocessor = None

# Load dataset for dropdown options
DATASET_PATH = os.path.join(BASE_DIR, "yield_df.csv")
df = pd.read_csv(DATASET_PATH)

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
    app.run(host="0.0.0.0", port=5050, debug=True)
