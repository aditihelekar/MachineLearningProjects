# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:46:02 2017

@author: Aditi
"""

# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd   
import os
from bs4 import BeautifulSoup   
import re
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and 
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    example1 = BeautifulSoup(raw_review)
    review_text = example1.get_text()
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #print(meaningful_words)
    #
    # 6. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join( meaningful_words ))   

def main():

    #os.chdir("E:\ComputerScience\Sem2\ML\ITCS6156_SLProject\AmazonReviews")    
    train = pd.read_csv("amazon_baby_train.csv")
    print(train["review"][167])
    print(train.shape)
    print("Size::" , train["review"].size)
    train = train.dropna(subset=['review'])
    print("Size::" , train["review"].size, train["rating"].size)
    clean_review = review_to_words( train["review"][0] )
    print( clean_review)
    # Get the number of reviews based on the dataframe column size
    num_reviews = train["review"].size

    # Initialize an empty list to hold the clean reviews
    clean_train_reviews = []

    # Loop over each review; create an index i that goes from 0 to the length
    # of the movie review list 
    for i in range(0, num_reviews):
        # If the index is evenly divisible by 1000, print a message
        #print(i)
        if( (i+1)%1000 == 0 ):
            print( "Review %d of %d\n" % ( i+1, num_reviews ))
        try:
            clean_train_reviews.append( review_to_words( train["review"][i] ))
        except KeyError:
            clean_train_reviews.append(" 0 ")
            print("Value not found", i)

        
    print( "Creating the bag of words...\n")


    # Initialize the "CountVectorizer" object, which is scikit-learn's
    # bag of words tool.  
    vectorizer = CountVectorizer(analyzer = "word",   \
                                 tokenizer = None,    \
                                 preprocessor = None, \
                                 stop_words = None,   \
                                 max_features = 5000) 
    
    # fit_transform() does two functions: First, it fits the model
    # and learns the vocabulary; second, it transforms our training data
    # into feature vectors. The input to fit_transform should be a list of 
    # strings.
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    
    # Numpy arrays are easy to work with, so convert the result to an 
    # array
    #train_data_features = train_data_features.toarray()
    print( train_data_features.shape)
    # Take a look at the words in the vocabulary
    vocab = vectorizer.get_feature_names()
    print( vocab)
    import numpy as np

    # Sum up the counts of each vocabulary word
    dist = np.sum(train_data_features, axis=0)
    
    # For each, print the vocabulary word and the number of times it 
    # appears in the training set
    for tag, count in zip(vocab, dist):
        print( count, tag)
        
    print( "Training the model...")
    
    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2),n_estimators=100,learning_rate=1)
    classifier.fit(train_data_features, train["rating"])
    
    # Read the test data
    test = pd.read_csv("amazon_baby_test.csv")
    test = test.dropna(subset=['review'])
    
    # Verify that there are 25,000 rows and 2 columns
    print (test.shape)
    
    # Create an empty list and append the clean reviews one by one
    num_reviews = len(test["review"])
    clean_test_reviews = [] 
    
    print( "Cleaning and parsing the test set movie reviews...\n")
    for i in range(0,num_reviews):
        if( (i+1) % 1000 == 0 ):
            print ("Review %d of %d\n" % (i+1, num_reviews))
        try:
            clean_review = review_to_words( test["review"][i] )
            clean_test_reviews.append( clean_review )
        except KeyError:
            clean_test_reviews.append(" 0 ")
            print("Value not found in test", i)
    
    # Get a bag of words for the test set, and convert to a numpy array
    test_data_features = vectorizer.transform(clean_test_reviews)
    #test_data_features = test_data_features.toarray()
    
    # Use the random forest to make sentiment label predictions
    result = classifier.predict(test_data_features)
	print(result)
    ratings = test["rating"]
	resultSize = len(result)
    print("Accuracy: ", accuracy_score(ratings[:resultSize], result))
            
    
main()



