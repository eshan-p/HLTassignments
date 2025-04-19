# Names (NetID): 
# Kenneth Anttila (kea200001)
# Aditya Desai    (amd210008)
# Tanmaye Goel    (txg220006)
# Eshan Patel     (exp200016)

# Section: CS 4395.001
# Movie Genre Prediction (Final Project)

import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def main():
    dataset_file = pd.read_csv('wiki_movie_plots_deduped.csv')
    corpus = preprocessing(dataset_file)
    datasets = encode_and_split(corpus)

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

main()