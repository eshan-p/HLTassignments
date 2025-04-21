# Names (NetID): 
# Kenneth Anttila (kea200001)
# Aditya Desai    (amd210008)
# Tanmaye Goel    (txg220006)
# Eshan Patel     (exp200016)

# Section: CS 4395.001
# Movie Genre Prediction (Final Project)

import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from gensim.models import Word2Vec
from keras.api.preprocessing.sequence import pad_sequences
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors



# Download NLTK data  --  only run once
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# only run this once to convert the file
# glove2word2vec(glove_input_file="glove.6B.300d.txt", word2vec_output_file="gensim_glove_vectors.txt")
glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)


# Instantiate lemmatizer
lemmatizer = WordNetLemmatizer()

# Define stopwords
stop_words = set(stopwords.words('english'))

# Load pre-trained Word2Vec model
# word2vec = gensim.models.KeyedVectors.load("word2vec-google-news-300.kv")

def main():
    dataset_file = pd.read_csv('wiki_movie_plots_deduped.csv')
    corpus = preprocessing(dataset_file)

    lemmatize(corpus)
    vectorize_plots(corpus)
    # wordEmbedding(corpus)

    datasets = encode_and_split(corpus)
    vectorized_train_plot = vectorization(datasets["train_plot"])
    vectorized_test_plot = vectorization(datasets["test_plot"])
    vectorized_val_plot = vectorization(datasets["val_plot"])

    print(corpus)
    print("Number of unique genres in dataset: ", corpus['Genre'].nunique())

    print("Number of entries in training set:", datasets["train_plot"].size)
    print("Number of entries in testing set:", datasets["test_plot"].size)
    print("Number of entries in validation set:", datasets["val_plot"].size)


def preprocessing(corpus):
    # remove all columns but "Genre" and "Plot"
    corpus = corpus.drop(["Release Year", "Title", "Origin/Ethnicity", "Director", "Cast", "Wiki Page"], axis='columns')

    # remove all rows with "unknown" genres
    corpus = corpus[corpus['Genre'] != 'unknown'].dropna(subset=['Plot', 'Genre'])

    # convert all entries to lower-case and remove punctuation from plot descriptions
    corpus['Plot'] = corpus['Plot'].str.lower().replace(r'[^\w\s]', '', regex=True)
    corpus['Genre'] = corpus['Genre'].str.lower()

    return corpus


def encode_and_split(corpus):
    le = LabelEncoder()
    corpus['Genre_Encoded'] = le.fit_transform(corpus['Genre']) # convert each genre label to a unique integer (ex. comedy = 573)

    plot = corpus['Plot']
    genre = corpus['Genre_Encoded']

    # split data into training, validation, and testing sets
    train_plot, x_plot, train_genre, x_genre = train_test_split(plot, genre, random_state=104, test_size=0.2, shuffle=True)
    val_plot, test_plot, val_genre, test_genre = train_test_split(x_plot, x_genre, random_state=104, test_size=0.5, shuffle=True)

    return {
            "train_plot": train_plot, 
            "train_genre": train_genre, 
            "val_plot": val_plot, 
            "val_genre": val_genre, 
            "test_plot": test_plot, 
            "test_genre": test_genre, 
           }

# Vectorization of input 
def vectorization(corpus):
    vectorizer = TfidfVectorizer()
    vectorized_plots = vectorizer.fit_transform(corpus)

    return vectorized_plots


def lemmatize(corpus):
    # lemmatize each word in the corpus
    lemmatized_corpus = []
    for plot in corpus['Plot']:
        words = plot.split()
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize words
        lemmatized_plot = ' '.join(words)
        lemmatized_corpus.append(lemmatized_plot)
    corpus['Plot'] = lemmatized_corpus
    # print(corpus)
    return lemmatized_corpus

# Word Embedding 
def vectorize_plots(corpus):
    print("Vectorizing plots...")
    vectorized_plots = []
    for plot in corpus['Plot']:
        words = plot.split()
        plot_matrix = []
        for word in words:
            if word in glove_model.key_to_index:
                plot_matrix.append(glove_model[word])
        plot_matrix = np.array(plot_matrix)
        vectorized_plots.append(plot_matrix)
    
    desc_len = 0
    for plot_matrix in vectorized_plots:
        desc_len += len(plot_matrix)
    desc_len /= int(len(vectorized_plots))
    desc_vectors = pad_sequences(vectorized_plots, maxlen=desc_len, padding='post', truncating='post', dtype='float32')



'''
def wordEmbedding(corpus):
    embeddingDictionary = {} 
    #load pretrained GloVe into dictionary 
    with open("glove.6B.300d.txt", encoding="utf8") as f: 
        for line in f: 
            values  = line.split() 
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddingDictionary[word] = vector
    plots = []
    for plot in corpus['Plot']: 
        words = plot.split()
        plot_matrix = []
        for word in words:
            if word in embeddingDictionary:
                vector = embeddingDictionary[word]
                plot_matrix.append(vector)
        # plot_matrix = np.array(plot_matrix)
        plots.append(plot_matrix)
    

    print(plots)

    desc_len = 0
    for plot_matrix in plots:
        desc_len += len(plot_matrix)
    desc_len /= int(len(plots))
    desc_vectors = pad_sequences(plots, maxlen=desc_len, padding='post', truncating='post', dtype='float32')

    # print(desc_vectors)
'''
# FFNN 

# Linear Regression 
# RNN 
# BERT 


main()