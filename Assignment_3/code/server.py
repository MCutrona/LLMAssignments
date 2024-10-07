# Using the flask framework to create a simple server
# that will cover the basics of deploying a machine learning model.

# neccessary imports
from flask import Flask, request, jsonify
import pickle
import numpy as np
from modelPickler import TrigramModel

# Flask app server
app = Flask(__name__)

# Load the pre-trained model
sentimentAnalysisModel = pickle.load(open('./models/sentimentAnalysisModel.pkl', 'rb'))
nGramModel = pickle.load(open('./models/nGramModel.pkl', 'rb'))

# Basic flask route at http://127.0.0.1:5000
@app.route('/')
def home():
    return 'Hello, World! This is the ML model API.'

# Flask route that shows that you can pass variables through the url
@app.route('/hello/<name>')
def hello_name(name):
    return f'Hello, {name}!'

@app.route('/sentimentAnalysis', methods=['POST'])
def sentimentAnalysis():
    # Get data from POST request
    data = request.get_json(force=True)
    print(data['features'])

    # Ensure that we received the expected array of features
    try:
        features = data['features']
    except KeyError:
        return jsonify(error="The 'features' key is missing from the request payload."), 400
    
    # Convert features into the right format and make a prediction
    predictions = [sentimentAnalysisModel.polarity_scores(feature) for feature in features]

    results = ['positive review' if prediction['compound'] > 0 else 'negative review' for prediction in predictions]
    
    # Return the prediction
    return jsonify(features=features, predictions=results)

@app.route('/nGram', methods=['POST'])
def nGram():
    # Get data from POST request
    data = request.get_json(force=True)
    print(data['features'])

    # Ensure that we received the expected array of features
    try:
        features = data['features']
    except KeyError:
        return jsonify(error="The 'features' key is missing from the request payload."), 400
    
    # Convert features into the right format and make a prediction
    predictions = [nGramModel.babble(feature[0], (feature[1], feature[2])) for feature in features]
    
    # Return the prediction
    return jsonify(features=features, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)