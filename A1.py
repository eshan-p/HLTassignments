# Names (NetID): 
# Kenneth Anttila (kea200001)
# Aditya Desai    (amd210008)
# Tanmaye Goel    (txg220006)
# Eshan Patel     (exp200016)

# Section: CS 4395.001
# Assignment 1

def main():
    file = open("A1_DATASET/train.txt", 'r')
    #test = file.read(); 
    #numWords = len(test.split()); 
    #print(numWords); 
    corpus = file.readlines()
    corpus = preprocessing(corpus)
    uProb, bProb = nGramMaker(corpus)
    print("Unigram Probabilities: \n ---------------")
    #for key in uProb: 
         #print(f"P({key}) = {uProb[key]}%")
    print("Bigram Probabilities: \n ---------------")
    for key in bProb: 
         print(f"P({key[1]} | {key[0]}) = {bProb[key]}%")
    

def preprocessing(corpus):
    text = [string.lower() for string in corpus]
    return text

"""
1. Unsmoothed n-grams
    - compute unigram & bigram probabilities from the training corpus
        - create matrix or dictionary of words
    - can only use libraries for preprocessing (corpus already tokenized tho)
        - each line in dataset files is a single review 
        - convert to lower-case --> lower() method
        - remove special characters(?) --> regex (re) library & sub() method
"""
def nGramMaker(corpus):
    unigramHolder = {} 
    bigramHolder = {} 
    unigramProbabilities = {} 
    bigramProbabilities = {} 
    
    for review in corpus: 
         tokens = review.split(' ')
         for i in range(len(tokens)): 
                unigramHolder[tokens[i]] = unigramHolder.get(tokens[i], 0) + 1
                if (i < len(tokens)-1):
                    bigram = (tokens[i],tokens[i+1])
                    bigramHolder[bigram] = bigramHolder.get(bigram, 0) + 1 
                    
    for key in unigramHolder: 
         unigramProbabilities[key] = (unigramHolder[key]/sum(unigramHolder.values()))*100
    for w1,w2 in bigramHolder:
         bigram = w1,w2
         bigramProbabilities[bigram] = (bigramHolder[bigram]/unigramHolder[w1])*100
    # for bigram in bigramHolder:
    #    bigramProbabilities[bigram] = bigramHolder[bigram]/unigramHolder[bigram[0]]

    return unigramProbabilities, bigramProbabilities

main()

"""
2. Smoothing & unknown words
    - 1) implement a method (or multiple methods) to handle unknown words
    - 2) implement the two smoothing methods (ex. Laplace & Add-k smoothing w/ different k)
"""


"""
3. Perplexity on Validation Set
    - compute the perplexity of the development/validation set --> val.txt
    - equation given in pdf
    - if we use more than one type of smoothing and unknown word handling, compare perplexity results in report
""" 
