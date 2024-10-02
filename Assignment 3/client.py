import requests
import pickle

# The URL for the predict route
url = 'http://127.0.0.1:5000/predict'

# Example input features
data = {
    'features': [5.1, 3.5, 1.4, 0.2]  # Example features for Iris dataset
}

# Send a POST request to the server
response = requests.post(url, json=data)

# Print the prediction result
print(response.json())
