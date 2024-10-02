# Author: Garrett Weaver

import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# NOTE: Uncomment the following lines to download the required NLTK resources
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('vader_lexicon')
# nltk.download('stopwords')

# Step 1: Load the data
data = pd.read_csv("movie.csv")

# Step 2: Clean and Prepare Text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Remove irrelevant characters and normalize text
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    words = word_tokenize(text)  # Tokenize into words
    words = [word for word in words if word not in stop_words]  # Remove stop words

    # remove br/<br> tags
    words = [word for word in words if word != 'br']

    return words

data['cleaned_text'] = data['text'].apply(clean_text)

# Extra: Method that gives context by providing the words surrounding the word of interest:
def get_context(review, word, window=5):
    for i, token in enumerate(review):
        if token == word:
            start = max(0, i - window)
            end = min(len(review), i + window + 1)
            return review[start:end]
    return []

word_of_interest = 'one'
# context = data['cleaned_text'].apply(get_context, word=word_of_interest)
# print(f"Context for the word '{word_of_interest}':")
# print(context.head())

# Step 4: Sentiment Analysis
# Using VADER sentiment analyzer for simplicity
sid = SentimentIntensityAnalyzer()

def get_sentiment_score(review):
    scores = sid.polarity_scores(' '.join(review))
    return scores['compound']  # Compound score represents overall sentiment

data['sentiment_score'] = data['cleaned_text'].apply(get_sentiment_score)

# Classifying the sentiment as positive, negative, or neutral
def classify_sentiment(score):
    if score > 0:
        return 'Positive'
    else:
        return 'Negative'

data['sentiment'] = data['sentiment_score'].apply(classify_sentiment)

# Compare the sentiment score with the sentiment label ("label") to see how we did give an accuracy score
data['label'] = data['label'].map({0: 'Negative', 1: 'Positive'})
accuracy = (data['sentiment'] == data['label']).mean()
print(f"Accuracy: {accuracy}")

# Show the first few rows of the dataset with sentiment
print("\nFirst few rows of the dataset with sentiment:")
print(data[['text', 'cleaned_text', 'sentiment_score', 'sentiment']].head())

# Step 3: Word Frequency Hunt
all_words = [word for review in data['cleaned_text'] for word in review]
word_counts = Counter(all_words)

# Visualize Word Frequencies (Word Cloud)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# Bar Chart for top N words
top_n_words = word_counts.most_common(20)
words, counts = zip(*top_n_words)
plt.figure(figsize=(10, 5))
plt.bar(words, counts)
plt.xticks(rotation=45)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top 20 Most Frequent Words')
plt.show()
