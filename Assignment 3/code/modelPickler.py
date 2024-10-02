# File that pickels models so that it can be used in the server
# source used to learn how to piclke models: https://medium.com/@spettiett/how-to-pickle-your-trained-model-f4b7051babaa

# neccessary imports (most needed to make the model)
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()

def pickle_model(model, filename):
    save_sid = open(filename, 'wb') # Open data file and write bytes
    pickle.dump(model, save_sid) # To store the result in a file
    save_sid.close() # Close data file

pickle_model(sid, 'models/model.pkl') # Pickle the model