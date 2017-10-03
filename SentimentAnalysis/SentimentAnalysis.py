# Import the pandas package, then use the "read_csv" function to read
# the labeled training data
import pandas as pd   
import os
from bs4 import BeautifulSoup   
import re
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")

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

      
    train = pd.read_csv("Cake_Review_Test_Data.csv")
    print(train.shape)
    print("Size::" , train["Review"].size)
    train = train.dropna(subset=['Review'])
    print("Size::" , train["Review"].size, train["Rating"].size)
    clean_review = review_to_words( train["Review"][0] )
    print( clean_review)
    # Get the number of reviews based on the dataframe column size
    num_reviews = train["Review"].size

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
            clean_train_reviews.append( review_to_words( train["Review"][i] ))
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
    

    print( train_data_features.shape)

    vocab = vectorizer.get_feature_names()
    print( vocab)
    import numpy as np

    dist = np.sum(train_data_features, axis=0)

    for tag, count in zip(vocab, dist):
        print( count, tag)
        
    print( "Training the model...")
    
    #classifier = KNeighborsClassifier(n_neighbors=2,weights='distance',algorithm='auto',leaf_size=5)
    #classifier = MultinomialNB(alpha=0.001)
    classifier = SVC(kernel='linear', probability=False, C=0.001)
    #classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=5),n_estimators=10,learning_rate=0.001)
    classifier.fit(train_data_features, train["Rating"])
    
    # Read the test data
    test = pd.read_csv("Train.csv")
    #test = test.dropna(subset=["Review"])
    
 
    print (test.shape)
    

    num_reviews = len(test["Review"])
    clean_test_reviews = [] 
    
    print( "Cleaning and parsing the test set movie reviews...\n")
    for i in range(0,num_reviews):
        if( (i+1) % 1000 == 0 ):
            print ("Review %d of %d\n" % (i+1, num_reviews))
        try:
            clean_review = review_to_words( test["Review"][i] )
            clean_test_reviews.append( clean_review )
        except KeyError:
            clean_test_reviews.append(" 0 ")
            print("Value not found in test", i)
    
    test_data_features = vectorizer.transform(clean_test_reviews)

    result = classifier.predict(test_data_features)
    print(result)
    ratings = test["Rating"]
    print("Accuracy:")
    print(accuracy_score(ratings, result))
    print("Confusion Matrix: ")
    print(confusion_matrix(ratings, result))
    print("Precision score: ")
    print(precision_score(result,ratings, average=None))
    print("Recall score: ")
    print(recall_score(result,ratings, average=None))
   
main()




