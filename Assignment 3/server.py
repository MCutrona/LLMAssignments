

from flask import Flask, request, jsonify
import pickle
import numpy as np
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(filename='server.log', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

# Load the pre-trained model
try:
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    app.logger.info('Model loaded successfully.')
except Exception as e:
    app.logger.error(f'Error loading model: {e}')
    model = None

@app.route('/')
def home():
    return 'Hello, World! This is the ML model API.'

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500
    try:
        data = request.get_json(force=True)
        if 'features' not in data:
            return jsonify({'error': "The 'features' key is missing from the request payload."}), 400
        features = data['features']
        if not isinstance(features, list):
            return jsonify({'error': "'features' should be a list of numerical values."}), 400
        if len(features) != 4:
            return jsonify({'error': 'Exactly 4 feature values are required.'}), 400
        try:
            features = [float(x) for x in features]
        except ValueError:
            return jsonify({'error': 'All feature values must be numeric.'}), 400
        prediction = model.predict([features])
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        app.logger.error(f'Exception during prediction: {e}')
        return jsonify({'error': 'An error occurred during prediction.'}), 500

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
