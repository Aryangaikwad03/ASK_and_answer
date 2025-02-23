from flask import Flask, request, jsonify
from flask_cors import CORS  # Enable CORS
import joblib  # Use joblib to load .pkl files
import numpy as np

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Absolute paths to your models
MALE_MODEL_PATH = 'Male_xgboost_model.pkl'  # Replace with the correct path
FEMALE_MODEL_PATH = 'Female_xgboost_model.pkl'  # Replace with the correct path

# Load your trained XGBoost models
male_model = joblib.load(MALE_MODEL_PATH)
female_model = joblib.load(FEMALE_MODEL_PATH)

# Function to preprocess input and make predictions
def predict_measurements(height, weight, gender):
    # Prepare input for the model (as a 2D array)
    input_data = np.array([[height, weight]])

    # Select the model based on gender
    if gender == 'male':
        model = male_model
    elif gender == 'female':
        model = female_model
    else:
        raise ValueError("Invalid gender. Must be 'male' or 'female'.")

    # Make predictions
    predictions = model.predict(input_data)

    # Debugging: Print the predictions to understand the output shape
    print("Raw Predictions:", predictions)

    # Ensure predictions is a 1D array
    if predictions.ndim == 2:
        predictions = predictions[0]  # Flatten to 1D array

    # Return predictions as a dictionary
    return {
        'chest': float(predictions[0]),  # Adjust indices based on your model's output
        'shoulder': float(predictions[1]),
        'waist': float(predictions[2]),
        'hips': float(predictions[3])
    }

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the frontend
    data = request.get_json()
    height = float(data['height'])
    weight = float(data['weight'])
    gender = data['gender']  # 'male' or 'female'

    # Call the function to get predictions
    predictions = predict_measurements(height, weight, gender)

    # Return predictions as JSON
    return jsonify(predictions)

if __name__ == '__main__':
    app.run(debug=True)