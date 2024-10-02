# Import necessary libraries
import nltk
import random
import math

from nltk.corpus import brown
from collections import Counter, defaultdict
from nltk.util import bigrams, trigrams
from nltk.lm import KneserNeyInterpolated, Laplace, MLE
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Vocabulary
from sklearn.model_selection import train_test_split

# Ensure the necessary NLTK data packages are downloaded
nltk.download('brown')
nltk.download('punkt')

# PART 1: Basic Probability in Language Modeling
# TASK 1: Calculate the frequency and probability of words in a given text (NLTK Brown News)

# Select words from the 'news' category
words = brown.words(categories='news')

# Calculate word frequencies
word_freq = Counter(words)

# Total number of words
total_words = len(words)

# Calculate word probabilities
word_prob = {word: freq / total_words for word, freq in word_freq.items()}

# Display the frequency and probability of the top 10 words
print("Word\tFrequency\tProbability")
for word, freq in word_freq.most_common(10):
    print(f"{word}\t{freq}\t{word_prob[word]:.6f}")

# PART 1: TASK 2: Calculate the entropy of a small text sample based on word probabilities

# Small text sample
sample_text = "The quick brown fox jumps over the lazy dog".split()

# Calculate entropy
entropy = -sum([
    word_prob.get(word, 1 / total_words) * math.log2(word_prob.get(word, 1 / total_words))
    for word in sample_text
])

print(f"\nEntropy of the sample text: {entropy:.4f} bits")

# PART 2: Building a Bigram Model
# TASK 1: Implement a simple bigram model using a given text corpus

# Create bigrams from the corpus
bigrams_list = list(bigrams(words))

# Calculate bigram frequencies
bigram_freq = Counter(bigrams_list)

# Calculate conditional probabilities
bigram_prob = defaultdict(dict)
for (w1, w2), freq in bigram_freq.items():
    bigram_prob[w1][w2] = freq / word_freq[w1]

# PART 2: TASK 2: Use the bigram model to generate sentences

def generate_sentence(bigram_prob, start_word, length=10):
    sentence = [start_word]
    for _ in range(length - 1):
        current_word = sentence[-1]
        next_words = bigram_prob.get(current_word)
        if not next_words:
            break  # Stop if there is no known continuation
        next_word = random.choices(
            population=list(next_words.keys()),
            weights=list(next_words.values())
        )[0]
        sentence.append(next_word)
    return ' '.join(sentence)

# Generate a sentence starting with 'The'
generated_sentence = generate_sentence(bigram_prob, 'The')
print(f"\nGenerated Sentence: {generated_sentence}")

# PART 3: Improving the N-gram Model
# TASK 1: Modify probability calculation to include Laplace smoothing and compare entropy

# Laplace smoothing parameters
vocab_size = len(word_freq)

# Apply Laplace smoothing to word probabilities
smoothed_word_prob = {
    word: (freq + 1) / (total_words + vocab_size)
    for word, freq in word_freq.items()
}

# Recalculate entropy with smoothed probabilities
smoothed_entropy = -sum([
    smoothed_word_prob.get(word, 1 / (total_words + vocab_size)) * math.log2(smoothed_word_prob.get(word, 1 / (total_words + vocab_size)))
    for word in sample_text
])

print(f"\nEntropy after Laplace smoothing: {smoothed_entropy:.4f} bits")

# TASK 2: Build and evaluate both bigram and trigram models using a larger dataset

# Load sentences from the 'news' category
sentences = brown.sents(categories='news')

# Split data into training and test sets
train_sentences, test_sentences = train_test_split(sentences, test_size=0.2, random_state=42)

# Build vocabulary from training data and include '<UNK>' token
def build_vocab(sentences):
    words = [word.lower() for sentence in sentences for word in sentence]
    vocab = Vocabulary(words, unk_cutoff=1)
    return vocab

vocab = build_vocab(train_sentences)

# Preprocess sentences to map OOV words to '<UNK>' and convert to lowercase
def preprocess_sentences(sentences, vocab):
    processed_sentences = []
    for sentence in sentences:
        processed_sentence = []
        for word in sentence:
            word_lower = word.lower()
            if word_lower in vocab:
                processed_sentence.append(word_lower)
            else:
                processed_sentence.append('<UNK>')
        processed_sentences.append(processed_sentence)
    return processed_sentences

# Preprocess training and test sentences
train_sentences_processed = preprocess_sentences(train_sentences, vocab)
test_sentences_processed = preprocess_sentences(test_sentences, vocab)

# Prepare training data using padded_everygram_pipeline
def prepare_lm_data(sentences, n):
    train_data, padded_sents = padded_everygram_pipeline(n, sentences)
    return train_data, padded_sents

# Bigram models
n_bigram = 2
train_data_bigram, padded_vocab_bigram = prepare_lm_data(train_sentences_processed, n_bigram)

# Trigram models
n_trigram = 3
train_data_trigram, padded_vocab_trigram = prepare_lm_data(train_sentences_processed, n_trigram)

# Build models with different smoothing techniques
# Kneser-Ney smoothing
kn_bigram_model = KneserNeyInterpolated(order=n_bigram, vocabulary=vocab)
kn_bigram_model.fit(train_data_bigram)

kn_trigram_model = KneserNeyInterpolated(order=n_trigram, vocabulary=vocab)
kn_trigram_model.fit(train_data_trigram)

# Laplace smoothing
laplace_bigram_model = Laplace(order=n_bigram, vocabulary=vocab)
laplace_bigram_model.fit(train_data_bigram)

laplace_trigram_model = Laplace(order=n_trigram, vocabulary=vocab)
laplace_trigram_model.fit(train_data_trigram)

# MLE (No smoothing)
mle_bigram_model = MLE(order=n_bigram, vocabulary=vocab)
mle_bigram_model.fit(train_data_bigram)

mle_trigram_model = MLE(order=n_trigram, vocabulary=vocab)
mle_trigram_model.fit(train_data_trigram)

# Evaluate models using perplexity
def calculate_model_perplexity(model, test_sentences):
    try:
        perplexity = model.perplexity(test_sentences)
        return perplexity
    except ZeroDivisionError:
        return float('inf')

# Compute perplexities for bigram models
perplexity_mle_bigram = calculate_model_perplexity(mle_bigram_model, test_sentences_processed)
perplexity_laplace_bigram = calculate_model_perplexity(laplace_bigram_model, test_sentences_processed)
perplexity_kn_bigram = calculate_model_perplexity(kn_bigram_model, test_sentences_processed)

print(f"\nBigram Model Perplexities:")
print(f"MLE Bigram Model Perplexity: {perplexity_mle_bigram}")
print(f"Laplace Bigram Model Perplexity: {perplexity_laplace_bigram}")
print(f"Kneser-Ney Bigram Model Perplexity: {perplexity_kn_bigram}")

# Compute perplexities for trigram models
perplexity_mle_trigram = calculate_model_perplexity(mle_trigram_model, test_sentences_processed)
perplexity_laplace_trigram = calculate_model_perplexity(laplace_trigram_model, test_sentences_processed)
perplexity_kn_trigram = calculate_model_perplexity(kn_trigram_model, test_sentences_processed)

print(f"\nTrigram Model Perplexities:")
print(f"MLE Trigram Model Perplexity: {perplexity_mle_trigram}")
print(f"Laplace Trigram Model Perplexity: {perplexity_laplace_trigram}")
print(f"Kneser-Ney Trigram Model Perplexity: {perplexity_kn_trigram}")

# Generate sentences using Kneser-Ney bigram model
def generate_sentence_kn(model, num_words, random_seed=42):
    random.seed(random_seed)
    content = []
    context = ('<s>',)
    for _ in range(num_words):
        word = model.generate(text_seed=context)
        if word == '</s>':
            break
        content.append(word)
        context = (word,)
    return ' '.join(content)

sentence_kn_bigram = generate_sentence_kn(kn_bigram_model, num_words=15)
print(f"\nGenerated Sentence with Kneser-Ney Bigram Model: {sentence_kn_bigram}")

# Generate sentences using Kneser-Ney trigram model
def generate_sentence_kn_trigram(model, num_words, random_seed=42):
    random.seed(random_seed)
    content = []
    context = ('<s>', '<s>')
    for _ in range(num_words):
        word = model.generate(text_seed=context)
        if word == '</s>':
            break
        content.append(word)
        context = (context[-1], word)
    return ' '.join(content)

sentence_kn_trigram = generate_sentence_kn_trigram(kn_trigram_model, num_words=15)
print(f"\nGenerated Sentence with Kneser-Ney Trigram Model: {sentence_kn_trigram}")
