# Using the flask framework to create a simple server that will cover the basics of deploying a machine learning model.

from flask import Flask, request, jsonify
import pickle
import numpy as np

# Flask app server
app = Flask(__name__)

# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))

# Basic flask route at http://127.0.0.1:5000
@app.route('/')
def home():
    return 'Hello, World! This is the ML model API.'

# Flask route that shows that you can pass variables through the url
@app.route('/hello/<name>')
def hello_name(name):
    return f'Hello, {name}!'

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from POST request
    data = request.get_json(force=True)

    # Ensure that we received the expected array of features
    try:
        features = data['features']
    except KeyError:
        return jsonify(error="The 'features' key is missing from the request payload."), 400
    
    # Convert features into the right format and make a prediction
    prediction = model.predict([features])

    # Return the prediction
    return jsonify(prediction=int(prediction[0]))
if __name__ == '__main__':
    app.run(debug=True)