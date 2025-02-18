# Names (NetID): 
# Kenneth Anttila (kea200001)
# Aditya Desai    (amd210008)
# Tanmaye Goel    (txg220006)
# Eshan Patel     (exp200016)

# Section: CS 4395.001
# Assignment 1

import math


def main():
    train_file = open("A1_DATASET/train.txt", 'r')
    
    # Preprocess the corpus and then train unsmoothed and smoothed unigram and bigram probabilities
    train_corpus = preprocessing(train_file.readlines())
    unigram_probs, bigram_probs = n_gram_maker(train_corpus)
    smoothed_unigram_probs, smoothed_bigram_probs = smoothed_n_gram_maker(train_corpus)

    print("Unigram Probabilities: \n ---------------")
    # for key in unigram_probs: 
    #      print(f"P({key}) = {unigram_probs[key]*100}%")
    print("Bigram Probabilities: \n ---------------")
    # for key in bigram_probs: 
    #     print(f"P({key[1]} | {key[0]}) = {bigram_probs[key]*100}%")

    # Print perplexities for unsmoothed and smoothed
    print("Unsmoothed perplexities: \n ---------------")
    validate(unigram_probs, bigram_probs)
    print("\nSmoothed perplexities: \n ---------------")
    validate(smoothed_unigram_probs, smoothed_bigram_probs)



def validate(unigram_probs, bigram_probs):
    test_file = open("A1_DATASET/val.txt", 'r')
    test_corpus = preprocessing(test_file.readlines())

    print("Unigram perplexity is ", ppl_unigram(test_corpus, unigram_probs))
    print("Bigram perplexity is ", ppl_bigram(test_corpus, bigram_probs))
    

def preprocessing(corpus):
    corpus = [(string.lower()).split() for string in corpus]
    return corpus   # Returns a list of lists (each inner list element is a word in a review, each review is an element in outer list)

"""
1. Unsmoothed n-grams
    - compute unigram & bigram probabilities from the training corpus
        - create matrix or dictionary of words
    - can only use libraries for preprocessing (corpus already tokenized tho)
        - each line in dataset files is a single review 
        - convert to lower-case --> lower() method
        - remove special characters(?) --> regex (re) library & sub() method
"""


def n_gram_maker(corpus):
    corpus = replace_unique_words(corpus)   # Handles unknown words by replacing unique words with "<unknown>"

    unigram_counts, unigram_probs = train_unigram(corpus)
    bigram_counts, bigram_probs = train_bigram(corpus, unigram_counts)

    return unigram_probs, bigram_probs

def smoothed_n_gram_maker(corpus):
    corpus = replace_unique_words(corpus)   # Handles unknown words by replacing unique words with "<unknown>"

    unigram_counts, unigram_probs = train_unigram(corpus)
    bigram_counts, bigram_probs = train_bigram(corpus, unigram_counts)

    return add_k_smoothing(unigram_counts, bigram_counts, 1) # Add-k smoothing the probabilities before returning


def train_unigram(corpus):
    unigram_counts = {}
    unigram_probs = {}

    for review in corpus:
        tokens = ["<start>"] + review + ["<stop>"]  # Add start and stop tokens for each review

        for unigram in tokens:
            unigram_counts[unigram] = unigram_counts.get(unigram, 0) + 1

    total_tokens = sum(unigram_counts.values()) # Total number of words/tokens in the corpus
    unigram_probs = {unigram: (count / total_tokens) for unigram, count in unigram_counts.items()}  # Calculates the probability for each unigram

    return unigram_counts, unigram_probs

def train_bigram(corpus, unigram_counts):
    bigram_counts = {}
    bigram_probs = {}

    for review in corpus:
         tokens = ["<start>"] + review + ["<stop>"]  # Add start and stop tokens for each review

         for bigram in zip(tokens, tokens[1:]): # Creates consecutive token pairs for the review, such as: (a, b), (b, c), (c, d)
             bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    bigram_probs = {bigram: (count / unigram_counts[bigram[0]]) for bigram, count in bigram_counts.items()} # Bigram counts are divided by unigram counts to get probability
    
    return bigram_counts, bigram_probs


"""
2. Smoothing & unknown words
    - 1) implement a method (or multiple methods) to handle unknown words
    - 2) implement the two smoothing methods (ex. Laplace & Add-k smoothing w/ different k)
"""
def replace_unique_words(corpus):
    word_counts = {}
    for review in corpus:
        for word in review:
            word_counts[word] = word_counts.get(word, 0) + 1
    
    return [["<unknown>" if word_counts[word] <= 1 else word for word in review] for review in corpus]  # Replace any word that occurs only once with unknown

def add_k_smoothing(unigram_counts, bigram_counts, k=1):
    total_tokens = sum(unigram_counts.values()) # Total number of tokens
    vocab_size = len(unigram_counts)    # Vocab size, total number of unigrams

    smoothed_unigram_probs = {unigram: ((count + k) / (total_tokens + k*vocab_size)) for unigram, count in unigram_counts.items()}
    smoothed_bigram_probs = {(first, second): ((bigram_counts.get((first, second), 0) + k) / (unigram_counts[first] + k*vocab_size)) for first in unigram_counts for second in unigram_counts}

    return smoothed_unigram_probs, smoothed_bigram_probs



"""
3. Perplexity on Validation Set
    - compute the perplexity of the development/validation set --> val.txt
    - equation given in pdf
    - if we use more than one type of smoothing and unknown word handling, compare perplexity results in report
""" 

def ppl_unigram(test_corpus, unigram_probs):
    ppl_sum = 0
    N = 0
    for review in test_corpus:
        tokens = ["<start>"] + review + ["<stop>"]
        N += len(tokens)
        for token in tokens:
            ppl_sum += -math.log(unigram_probs.get(token, unigram_probs["<unknown>"]))

    ppl_sum /= N
    print("ppl_sum: ", ppl_sum)
    return math.exp(ppl_sum)

def ppl_bigram(test_corpus, bigram_probs):
    ppl_sum = 0
    N = 0
    for review in test_corpus:
        tokens = ["<start>"] + review + ["<stop>"]
        N += len(tokens)

        for bigram in zip(tokens, tokens[1:]):

            # Not sure about this part, it's unclear which of these bigrams to default to and in what order
            # Especially with add-k smoothing
            if bigram_probs.get(bigram) is not None: probability = bigram_probs.get(bigram)
            elif bigram_probs.get((bigram[0], "<unknown>")) is not None: probability = bigram_probs.get((bigram[0], "<unknown>"))
            elif bigram_probs.get(("<unknown>", bigram[1])) is not None: probability = bigram_probs.get(("<unknown>", bigram[1]))
            elif bigram_probs.get(("<unknown>", "<unknown>")) is not None: probability = bigram_probs.get(("<unknown>", "<unknown>"))
            else: probability = 0    # Will raise error but should not be possible to reach anyways

            ppl_sum += -math.log(probability)

    ppl_sum /= N
    print("ppl_sum: ", ppl_sum)
    return math.exp(ppl_sum)


main()