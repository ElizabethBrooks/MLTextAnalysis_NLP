# Author: Elizabeth Brooks, Hayden Fuss
# Adapted from: relevanceclassifier.py
# Date Modified: 07/12/2015

# PreProcessor Directives
import os
import inspect
import sys
import csv
import yaml
import re
import random
sys.path.append(os.path.realpath('../'))
import twittercriteria as twc
# Classification function imports
import nltk
from nltk.classify import apply_features
from nltk.classify import SklearnClassifier
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer

# Global field declarations
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

# Size of chi2 sample, needs to be tuned for best results with MNB

# Define class to classify tweet relevance
class SentimentClassifier(object):
    # Class constructor to initialize classifier
    def __init__(self, scared_path='/scaredTraining.txt', angry_path='/angryTraining.txt',
                 sad_path='/sadTraining.txt', cheerful_path='/cheerfulTraining.txt',
                 objective_path='/objectiveTraining.txt'):
        # Initialize data sets
        self.trainingSet = [] # Labeled tweet training data set 
        self.labeledTweets = [] # Feature set of labeled tweet terms
        self.allTerms = []      
        # Begin functions for classification
        self.initLabeledSet(scared_path, angry_path, sad_path, cheerful_path, objective_path) # Initialize labeledTweets
        self.initTrainingSet() # Initialize trainingSet
        self.trainClassifier()
        # End func return
        return
    # End class constructor
    
    # Function to initialize the feature sets
    def initLabeledSet(self, scared_path, angry_path, sad_path, cheerful_path, objective_path):
        # Loop through the txt files line by line
        # Assign labels to tweets
        # Two classes, relevant and irrelevant
        paths = {'scared':scared_path, 'angry':angry_path, 'sad':sad_path,
                 'cheerful':cheerful_path, 'objective':objective}
        for sentiment in paths.keys():
            with open(current_dir + paths[sentiment], "r") as trainingFile:
                for line in trainingFile:
                    self.labeledTweets.append((line.split(), sentiment))
        # Randomize the data
        random.shuffle(self.labeledTweets)
        # End func return
        return
    # End initDictSet
    
    # Function for initializing the labeled training data set
    def initTrainingSet(self):
        self.getTweetText()
        self.getTerms()
        # The apply_features func processes a set of labeled tweet strings using the passed extractFeatures func
        self.trainingSet = apply_features(self.extractFeatures, self.labeledTweets)
        # End func return
        return
    # End initTrainingSet
    
    # Function to get relevant tweet terms
    def getTweetText(self):
        for (terms, sentiment) in self.labeledTweets:
            self.allTerms.extend(terms)
        # End for
        # End func return
        return
    # End getTweetText

    # Function to get term features
    def getTerms(self):
        self.allTerms = nltk.FreqDist(self.allTerms)
        self.wordFeatures = self.allTerms.keys()
        # End func return
        return
    # End getTerms

    # Function to extract features from tweets
    def extractFeatures(self, tweet_terms):
        tweet_terms = set(tweet_terms)
        # Set of unique tweet terms
        tweetFeatures = {}
        for word in self.wordFeatures:
            tweetFeatures['contains(%s)' % word] = (word in tweet_terms)
        # End for
        # Return feature set
        return tweetFeatures
    # End extractFeatures
    
    # Function to train the input NB classifier using it's apply_features func
    # Should be overridden by child classes
    def trainClassifier(self):
        self.classifier = nltk.NaiveBayesClassifier.train(self.trainingSet)
        # End func return
        return
    # End trainClassifier

    # Function to classify input tweet  
    def classify(self, tweet_text):
        return self.classifier.classify(self.extractFeatures(twc.cleanForSentiment(tweet_text).split()))

    # def test(self, testSetRel, testSetIrr):
    #     for each in testSetRel, testSetIrr:
    #         for i in range(0, len(each)):
    #             each[i] = self.extractFeatures(twc.cleanUpTweet(each[i]).split())
    #     rel = np.array(self.classifier.classify_many(testSetRel))
    #     irr = np.array(self.classifier.classify_many(testSetIrr))   
    #     self.tp = (rel == 'relevant').sum()
    #     self.fn = (rel == 'irrelevant').sum()
    #     self.fp = (irr == 'relevant').sum()
    #     self.tn = (irr == 'irrelevant').sum()
    #     return

    def balancedF(self):
        prec = float(self.tp)/(float(self.tp) + float(self.fp))
        recall = float(self.tp)/(float(self.tp) + float(self.fn))
        Fscore = 2*prec*recall/(prec + recall)
        return Fscore

    # def confusionMatrix(self):
    #     print "Confusion matrix:\n%d\t%d\n%d\t%d" % (self.tp, self.fn, self.fp, self.tn)
    #     return

# End class    

# Sub class to weight term relevance and use Bag-Of-Words (MultinomialNB)
class SentimentMNB(SentimentClassifier):
    # Sub class constructor
    def __init__(self, chiK=3368):
        # Call the super class constructor which initializes the classifier
        self.chiK = chiK
        super(SentimentMNB, self).__init__()
        # End func return
        return
    # End wrapper class constructor
    
    # Function to initialize the classifier pipeline
    def initPipeline(self):
        # Pipeline of transformers with a final estimator
        # The pipeline class behaves like a compound classifier
        # pipeline(steps=[...])

        # Old MNB pipeline with TFIDF
        # self.pipeline = Pipeline([('tfidf', TfidfTransformer()),
        #              ('chi2', SelectKBest(chi2, k=1000)),
        #              ('nb', MultinomialNB())])

        self.pipeline = Pipeline([('chi2', SelectKBest(chi2, k=self.chiK)),
                      ('nb', MultinomialNB())])
        # End func return
        return
    # End initPipeline
        
    # Overriding func to train multinomial NB classifier
    def trainClassifier(self):
        self.initPipeline()
        # Create the multinomial NB classifier
        self.classifier = SklearnClassifier(self.pipeline)
        # Train the classifier
        self.classifier.train(self.trainingSet)
        # End func return
        return
    # End trainClassifier override
# End sub class