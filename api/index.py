import pandas as pd
import joblib
from flask import Flask, request, jsonify

# Initialize App: Create a new Flask application.
app = Flask(__name__)

# Load Model: When the app starts, it immediately loads the .pkl file we created.
# This is a one-time process, making the app fast.
try:
    model = joblib.load('insurance_rf_model.pkl')
except FileNotFoundError:
    print("Error: 'insurance_rf_model.pkl' not found.")
    model = None

# Define the API endpoint: This is the special URL where users can send requests.
# It only accepts POST requests with a JSON body.
@app.route('/predict', methods=['POST'])
def predict():
    # Get Data: Grab the data that the user sent to us.
    data = request.get_json()

    # Check for Errors: Make sure the user sent all the required information.
    required_fields = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Missing required fields in request.'}), 400

    try:
        # Predict: Use our loaded model to make a prediction based on the user's data.
        user_df = pd.DataFrame([data])
        prediction = model.predict(user_df)[0]
        
        # Send Response: Return the prediction as a nicely formatted JSON response.
        return jsonify({'premium': round(prediction, 2)})
    except Exception as e:
        # Handle Errors: If something goes wrong, we send an error message instead of crashing.
        return jsonify({'error': str(e)}), 500

