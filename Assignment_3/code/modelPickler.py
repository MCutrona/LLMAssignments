# File that pickels models so that it can be used in the server
# source used to learn how to piclke models: https://medium.com/@spettiett/how-to-pickle-your-trained-model-f4b7051babaa

# neccessary imports (most needed to make the model)
import pickle
import nltk
import random
from nltk.corpus import brown
from nltk.sentiment import SentimentIntensityAnalyzer

# Load the pre-trained sentiment analysis model
sid = SentimentIntensityAnalyzer()

# Load the pre-trained nGram model
# Define a class for the trigram model using Markov chains
class TrigramModel:

    def __init__(self):
        self.memory = {}

    def learn_key(self, key, value):
        if key not in self.memory:
            self.memory[key] = []

        self.memory[key].append(value)
    
    def learn(self, text):
        tokens = text.split()
        trigrams = [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)]
        for trigram in trigrams:
            self.learn_key((trigram[0], trigram[1]), trigram[2])
        
    def next(self, current_state): # current_state is a tuple ('word1', 'word2')
        next_possible = self.memory.get(current_state)

        if not next_possible:
            next_possible = list(self.memory.keys())
            # Get a random value from a random key and return it
            return random.sample(next_possible, 1)[0][1]
        
        return random.sample(next_possible, 1)[0]
    
    def babble(self, amount, state=('', '')):
        if not amount:
            return state[0] + ' ' + state[1]
        
        next_word = self.next(state)

        if not next_word:
            return state[0] + ' ' + state[1]
        
        return state[0] + ' ' + self.babble(amount - 1, (state[1], next_word))

    
all_words = brown.words()
triModel = TrigramModel()
triModel.learn(' '.join(list(all_words)))

def pickle_model(model, filename):
    save_sid = open(filename, 'wb') # Open data file and write bytes
    pickle.dump(model, save_sid) # To store the result in a file
    save_sid.close() # Close data file

pickle_model(sid, 'models/sentimentAnalysisModel.pkl') # Pickle the model
pickle_model(triModel, 'models/nGramModel.pkl') # Pickle the model