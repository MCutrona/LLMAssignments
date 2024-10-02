# given code for a python based client

# neccessary imports
import requests

# The URL for the predict route
url = 'http://127.0.0.1:5000/sentimentAnalysis'

# Example input features
data = {'features': ['I hate this movie.  Its terrible', 'this movie is awesome!']} # Example features for Iris dataset (These are for demonstration purposes only CHANGE THE VALUES)

# Send a POST request to the server
response = requests.post(url, json=data)

# Print the prediction result
print(response.json())

# You can also test the Flask server using a curl command from the terminal. Here's how you can do it:
# curl -X POST -H "Content-Type: application/json" -d '{"features":[5.1, 3.5, 1.4, 0.2]}' http://127.0.0.1:5000/predict
# This sends a POST request to the /predict route with the same example input features as the Python client.