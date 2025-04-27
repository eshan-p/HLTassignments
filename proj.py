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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from keras.api.preprocessing.sequence import pad_sequences
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors
from keras.api.models import Sequential
from keras.api.layers import Dense, Dropout, Flatten, Embedding, SimpleRNN
from keras.api.callbacks import EarlyStopping
from keras.api.utils import to_categorical


# Download NLTK data  --  only run once
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# only run this once to convert the file
# converts GloVe format to word2vec format
# glove2word2vec(glove_input_file="glove.6B.300d.txt", word2vec_output_file="gensim_glove_vectors.txt")
# load the word2vec format GloVe model
glove_model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)


# Instantiate lemmatizer
lemmatizer = WordNetLemmatizer()

# Define stopwords
stop_words = set(stopwords.words('english'))

def main():
    # read in dataset and basic preprocessing
    dataset_file = pd.read_csv('wiki_movie_plots_deduped.csv')
    corpus = preprocessing(dataset_file)

    # standardize genre labels and remove rare genres
    corpus = normalize_genre_labels(corpus)
    corpus = filter_rare_genres(corpus, min_count=50)

    # lemmatize plot descriptions and vectorize them
    lemmatize(corpus)
    vectorized_plots = vectorize_plots(corpus)

    # encode genre labels and split data into training, validation, and testing sets
    datasets = encode_and_split(corpus, vectorized_plots)

    # print corpus and genre set for info
    print(corpus)
    genre_set = set(corpus['Genre'])

    print("Unique genres:", genre_set)
    print("Number of unique genres in dataset: ", corpus['Genre'].nunique())
    
    # number of output classes (genres)
    num_classes = corpus['Genre_Encoded'].nunique()

    # ffnn model
    ffnn_model = train_ffnn_model(
        datasets["train_plot"], datasets["train_genre"],
        datasets["val_plot"], datasets["val_genre"],
        num_classes
    )

    # rnn model
    rnn_model = RNN(datasets["train_plot"], datasets["train_genre"], 
                    datasets["val_plot"], datasets["val_genre"], 
                    num_classes) 

    # regression models
    linear_model, logistic_model = regression_models(datasets["train_plot"], datasets["train_genre"], 
                                        datasets["val_plot"], datasets["val_genre"])
    
    # evaluate models on test set
    test_genre_int = datasets["test_genre"].copy()
    datasets["test_genre"] = to_categorical(datasets["test_genre"], num_classes)
    X_test_flat = np.mean(datasets["test_plot"], axis=1)

    
    # evaluate models and store results
    results = []
    results.append({"Model": "Feedforward Neural Network", 
                    "Accuracy": ffnn_model.evaluate(datasets["test_plot"], datasets["test_genre"])[1]})
    results.append({"Model": "Recurrent Neural Network", 
                    "Accuracy": rnn_model.evaluate(datasets["test_plot"], datasets["test_genre"])[1]}) 
    results.append({"Model": "Linear Regression", 
                    "Accuracy": linear_model.score(X_test_flat, test_genre_int)})
    results.append({"Model": "Logistic Regression", 
                    "Accuracy": logistic_model.score(X_test_flat, test_genre_int)})

    # print results
    results_df = pd.DataFrame(results)

    print("\nModel Performance Results:")
    print("===================================")
    print(results_df)


def preprocessing(corpus):
    # remove all columns but "Genre" and "Plot"
    corpus = corpus.drop(["Release Year", "Title", "Origin/Ethnicity", "Director", "Cast", "Wiki Page"], axis='columns')

    # remove all rows with "unknown" genres
    corpus = corpus[corpus['Genre'] != 'unknown'].dropna(subset=['Plot', 'Genre'])

    # convert all entries to lower-case and remove punctuation from plot descriptions
    corpus['Plot'] = corpus['Plot'].str.lower().replace(r'[^\w\s]', '', regex=True)
    corpus['Genre'] = corpus['Genre'].str.lower()

    return corpus

def filter_rare_genres(corpus, min_count=10):
    """
    Removes movies with genres that appear less than `min_count` times.
    """
    genre_counts = corpus['Genre'].value_counts()
    frequent_genres = genre_counts[genre_counts >= min_count].index
    filtered_corpus = corpus[corpus['Genre'].isin(frequent_genres)].reset_index(drop=True)
    print(f"Filtered out rare genres. Remaining genres: {len(frequent_genres)}")
    return filtered_corpus


def normalize_genre_labels(corpus):
    """
    Standardizes genre strings by:
    - Replacing slashes, ampersands, etc. with commas
    - Removing extra whitespace
    - Sorting genre components alphabetically
    - Rejoining them with a single space
    """
    def normalize(genre_str):
        # Lowercase and replace separators with commas
        cleaned = re.sub(r'[\/&-]', ',', genre_str.lower())
        # Split into individual genres, strip whitespace
        parts = [part.strip() for part in cleaned.split(',') if part.strip()]
        # Sort genre components alphabetically and join with space
        return ' '.join(sorted(parts))

    corpus['Genre'] = corpus['Genre'].apply(normalize)
    return corpus

def encode_and_split(corpus, vectorized_plots):
    le = LabelEncoder()
    # convert each genre label to a unique integer (ex. comedy = 573)
    corpus['Genre_Encoded'] = le.fit_transform(corpus['Genre'])

    genre = corpus['Genre_Encoded']

    # split data into training, validation, and testing sets
    train_plot, x_plot, train_genre, x_genre = train_test_split(vectorized_plots, genre, random_state=104, test_size=0.2, shuffle=True)
    val_plot, test_plot, val_genre, test_genre = train_test_split(x_plot, x_genre, random_state=104, test_size=0.5, shuffle=True)

    return {
            "train_plot": train_plot, 
            "train_genre": train_genre, 
            "val_plot": val_plot, 
            "val_genre": val_genre, 
            "test_plot": test_plot, 
            "test_genre": test_genre, 
           }

def lemmatize(corpus):
    # lemmatize each word in the corpus
    lemmatized_corpus = []
    for plot in corpus['Plot']:
        words = plot.split()
        words = [word for word in words if word not in stop_words]  # Remove stopwords
        words = [lemmatizer.lemmatize(word) for word in words]  # Lemmatize words
        lemmatized_plot = ' '.join(words) # Join words back into a string
        lemmatized_corpus.append(lemmatized_plot) # Store lemmatized plot
    # update corpus with lemmatized plots
    corpus['Plot'] = lemmatized_corpus
    return lemmatized_corpus

# Word Embedding 
def vectorize_plots(corpus):
    print("Vectorizing plots...")
    vectorized_plots = []
    for plot in corpus['Plot']:
        words = plot.split()
        # replace word with its GloVe vector if it exists in the model
        # otherwise, ignore the word
        plot_matrix = np.array([glove_model[word] for word in words if word in glove_model.key_to_index])
        vectorized_plots.append(plot_matrix)

    # Pads based on average length of sequence
    # TODO: maybe try median instead?
    desc_len = 0
    for plot_matrix in vectorized_plots:
        desc_len += len(plot_matrix)
    # divide by number of plots to get average length
    desc_len = int(desc_len / len(vectorized_plots))
    print("Average plot length", desc_len)
    # pad sequences to the same length
    desc_vectors = pad_sequences(vectorized_plots, maxlen=desc_len, padding='post', truncating='post', dtype='float32')

    return desc_vectors

# FFNN 
def train_ffnn_model(train_X, train_y, val_X, val_y, num_classes):
    print("Training Feedforward Neural Network...")

    # Convert labels to categorical format (one-hot encoding)
    train_y_cat = to_categorical(train_y, num_classes=num_classes)
    val_y_cat = to_categorical(val_y, num_classes=num_classes)

    # Define the FFNN model
    model = Sequential()
    model.add(Flatten(input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))

    # compiling Tensorflow model created above
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model
    history = model.fit(train_X, train_y_cat, validation_data=(val_X, val_y_cat), epochs=10, batch_size=64, callbacks=[early_stop])

    print("Model training complete.")

    return model

# RNN 
def RNN(train_X, train_Y, val_X, val_Y, num_classes):
    print("Training Recurrent Neural Network...")

    # creating RNN architecture and adding layers
    model = Sequential()
    model.add(SimpleRNN(64, return_sequences=True)) # could substitute SimpleRNN for LSTM here
    model.add(SimpleRNN(64))
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation="sigmoid"))

    # compiling Tensorflow model created above
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

    train_Y_cat = to_categorical(train_Y, num_classes=num_classes)
    val_Y_cat = to_categorical(val_Y, num_classes=num_classes)

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

    # Train the model
    history = model.fit(train_X, train_Y_cat, validation_data=(val_X, val_Y_cat), epochs=10, batch_size=64, callbacks=[early_stop])
                        
    return model


# Compare and return best regression model (Linear vs. Logistic)
def regression_models(train_X, train_Y, val_X, val_Y):
    # flatten 3D vectors for 2D regression models
    train_X_flat = np.mean(train_X, axis=1)
    val_X_flat = np.mean(val_X, axis=1)

    # linear regression model
    linear_model = LinearRegression()
    linear_model.fit(train_X_flat, train_Y)
    linear_Y_pred = linear_model.predict(val_X_flat)
    linear_Y_pred_rounded = np.round(linear_Y_pred).astype(int)
    linear_Y_pred_rounded = np.clip(linear_Y_pred_rounded, 0, max(train_Y.max(), val_Y.max()))
    linear_accuracy = accuracy_score(val_Y, linear_Y_pred_rounded)

    # logistic regression model
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(train_X_flat, train_Y)
    logistic_Y_pred = logistic_model.predict(val_X_flat)
    logistic_accuracy = accuracy_score(val_Y, logistic_Y_pred)

    return linear_model, logistic_model

main()