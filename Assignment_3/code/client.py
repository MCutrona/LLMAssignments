# given code for a python based client
# https://8d3e-35-196-52-251.ngrok-free.app/
# neccessary imports
import requests

# The URL for the predict route
SentimentUrl = 'http://127.0.0.1:5000/sentimentAnalysis'
nGramUrl = 'http://127.0.0.1:5000/nGram'

# Example input features
SentimentData = {'features': ['I hate this movie.  Its terrible', 'this movie is awesome!']} # Example features for Sentiment Analysis
nGramData = {'features': [[25, 'the', 'quick']]} # Example features for nGram

# Send a POST request to the server
response1 = requests.post(SentimentUrl, json=SentimentData)
response2 = requests.post(nGramUrl, json=nGramData)

# Print the prediction result
print(response1.json())
print(response2.json())

# You can also test the Flask server using a curl command from the terminal. Here's how you can do it:
# curl -X POST -H "Content-Type: application/json" -d '{"features":[5.1, 3.5, 1.4, 0.2]}' http://127.0.0.1:5000/predict
# This sends a POST request to the /predict route with the same example input features as the Python client.