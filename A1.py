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
    
    u_probs, b_probs = basic_n_gram_maker(train_corpus)
    unigram_probs, bigram_probs = n_gram_maker(train_corpus)
    add1_unigram_probs, add1_bigram_probs = smoothed_n_gram_maker(train_corpus, 'add1')
    addk_unigram_probs, addk_bigram_probs = smoothed_n_gram_maker(train_corpus, 'addk')

    print("Unigram Probabilities: \n ---------------")
    for i, key in enumerate(unigram_probs):
        if i == 12: 
            break
        print(f"P({key}) = {unigram_probs[key]*100}%")

    print("\nBigram Probabilities: \n ---------------")
    for i, key in enumerate(bigram_probs): 
        if i == 12:
            break
        print(f"P({key[1]} | {key[0]}) = {bigram_probs[key]*100}%")

    # Print perplexities for unsmoothed and smoothed
    print("\nUnsmoothed perplexities: \n ---------------")
    validate(u_probs, b_probs)

    print("\nUnsmoothed perplexities with unknown word handling: \n ---------------")
    validate(unigram_probs, bigram_probs)

    print("\nAdd 1 Smoothed perplexities: \n ---------------")
    validate(add1_unigram_probs, add1_bigram_probs)

    print("\nAdd k Smoothed perplexities: \n ---------------")
    validate(addk_unigram_probs, addk_bigram_probs)



def validate(unigram_probs, bigram_probs):
    test_file = open("A1_DATASET/val.txt", 'r')
    test_corpus = preprocessing(test_file.readlines())

    print("Unigram perplexity is ", ppl_unigram(test_corpus, unigram_probs))
    print("Bigram perplexity is ", ppl_bigram(test_corpus, bigram_probs))
    

def preprocessing(corpus):
    corpus = [(string.lower()).split() for string in corpus]    # Replace the corpus as list of lists
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

# N gram maker without handling unknown words or smoothing
def basic_n_gram_maker(corpus):
    unigram_counts, unigram_probs = train_unigram(corpus)
    bigram_counts, bigram_probs = train_bigram(corpus, unigram_counts)
    return unigram_probs, bigram_probs

# N gram maker that handles unknown words but doesn't implement any smoothing
def n_gram_maker(corpus):
    corpus = replace_unique_words(corpus)   # Handles unknown words by replacing unique words with "<unknown>"

    unigram_counts, unigram_probs = train_unigram(corpus)
    bigram_counts, bigram_probs = train_bigram(corpus, unigram_counts)

    return unigram_probs, bigram_probs

# N gram maker that both handles unknown words and uses add-1 or add-k smoothing
def smoothed_n_gram_maker(corpus, smoothing='add1'):
    corpus = replace_unique_words(corpus)   # Handles unknown words by replacing unique words with "<unknown>"

    unigram_counts, unigram_probs = train_unigram(corpus)
    bigram_counts, bigram_probs = train_bigram(corpus, unigram_counts)

    # Two different smoothing types
    if smoothing == 'add1':
        return add_one_smoothing(unigram_counts, bigram_counts) # Add-one smoothing the probabilities before returning
    elif smoothing == 'addk':
        return add_k_smoothing(unigram_counts, bigram_counts, 0.05) # Add-k smoothing the probabilities before returning
    else: return

# Method to train a unigram given corpus
def train_unigram(corpus):
    unigram_counts = {}
    unigram_probs = {}

    for review in corpus:
        tokens = ["<start>"] + review + ["<stop>"]  # Add start and stop tokens for each review

        for unigram in tokens:
            unigram_counts[unigram] = unigram_counts.get(unigram, 0) + 1

    total_tokens = sum(unigram_counts.values()) # Total number of words/tokens in the corpus
    unigram_probs = {unigram: (count / total_tokens) for unigram, count in unigram_counts.items()}  # Calculates the probability for each unigram

    return unigram_counts, unigram_probs    # Return both frequency and probability dicts

def train_bigram(corpus, unigram_counts):
    bigram_counts = {}
    bigram_probs = {}

    for review in corpus:
         tokens = ["<start>"] + review + ["<stop>"]  # Add start and stop tokens for each review

         for bigram in zip(tokens, tokens[1:]): # Creates consecutive token pairs for the review, such as: (a, b), (b, c), (c, d)
             bigram_counts[bigram] = bigram_counts.get(bigram, 0) + 1

    bigram_probs = {bigram: (count / unigram_counts[bigram[0]]) for bigram, count in bigram_counts.items()} # Bigram counts are divided by unigram counts to get probability
    
    return bigram_counts, bigram_probs     # Return both frequency and probability dicts


"""
2. Smoothing & unknown words
    - 1) implement a method (or multiple methods) to handle unknown words
    - 2) implement the two smoothing methods (ex. Laplace & Add-k smoothing w/ different k)
"""

# Method to handle unknown words by replacing all words that appear once or fewer with <unknown>
def replace_unique_words(corpus):
    word_counts = {}
    for review in corpus:
        for word in review:
            word_counts[word] = word_counts.get(word, 0) + 1    # Counts frequency of each word in the corpus
    
    return [["<unknown>" if word_counts[word] <= 1 else word for word in review] for review in corpus]  # Replace any word that occurs only once with unknown

# Add one smoothing method, adds 1 to all counts and V (vocab size) to denominator when calculating probability
def add_one_smoothing(unigram_counts, bigram_counts):
    total_tokens = sum(unigram_counts.values()) # Total number of tokens
    vocab_size = len(unigram_counts)    # Vocab size, total number of unigrams

    smoothed_unigram_probs = {unigram: ((count + 1) / (total_tokens + vocab_size)) for unigram, count in unigram_counts.items()}
    smoothed_bigram_probs = {(first, second): ((bigram_counts.get((first, second), 0) + 1) / (unigram_counts[first] + vocab_size)) for first in unigram_counts for second in unigram_counts}

    return smoothed_unigram_probs, smoothed_bigram_probs

# Add k smoothing method, adds k to all counts and k*V (vocab size) to denominator when calculating probability
def add_k_smoothing(unigram_counts, bigram_counts, k):
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

# Perplexity calculation for unigrams
def ppl_unigram(test_corpus, unigram_probs):
    ppl_sum = 0
    N = 0

    # Sum perplexity for each token in the corpus
    for review in test_corpus:
        tokens = ["<start>"] + review + ["<stop>"]
        N += len(tokens)    # Tracking total number of tokens
        for token in tokens:
            ppl_sum += -math.log(unigram_probs.get(token, unigram_probs.get("<unknown>", 1)))   # Defaults to p(unknown) and then 1 which adds 0 to the sum

    ppl_sum /= N
    return math.exp(ppl_sum)

# Perplexity calculation for bigrams
def ppl_bigram(test_corpus, bigram_probs):
    ppl_sum = 0
    N = 0

    # Sum perplexity for each token in the corpus
    for review in test_corpus:
        tokens = ["<start>"] + review + ["<stop>"]
        N += len(tokens)    # Tracking total number of tokens

        for bigram in zip(tokens, tokens[1:]):
            # Tries to find bigram, and then replaces one or both tokens with <unknown>
            # Defaults to 1 (which adds 0 to the sum) in case unknown words aren't handled
            if bigram_probs.get(bigram) is not None: probability = bigram_probs.get(bigram)
            elif bigram_probs.get((bigram[0], "<unknown>")) is not None: probability = bigram_probs.get((bigram[0], "<unknown>"))
            elif bigram_probs.get(("<unknown>", bigram[1])) is not None: probability = bigram_probs.get(("<unknown>", bigram[1]))
            elif bigram_probs.get(("<unknown>", "<unknown>")) is not None: probability = bigram_probs.get(("<unknown>", "<unknown>"))
            else: probability = 1    # Ignores this token, (log(1) == 0)

            ppl_sum += -math.log(probability)

    ppl_sum /= N
    return math.exp(ppl_sum)


main()