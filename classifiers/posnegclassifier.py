# Author: Elizabeth Brooks
# File: posnegclassifier.py
# Date Modified: 08/03/2015
# Edited: Hayden Fuss

# Begin script

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
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn import metrics

# Global field declarations
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

##########################################################################################################################

# Define class to classify tweet relevance
class PosNegClassifier(object):
    # Class constructor to initialize classifier
    def __init__(self, paths, cleaner):
        self.cleaner = cleaner
        # Initialize data sets
        self.categories = [] # Feature/Term, category/class set
        self.tweets = [] # Tweet text/feature strings
        self.labels = [] # Tweet categories/classes
        # Begin functions for classification
		# Initialize classes using input txt file paths
        self.initCategories(paths)
        # Initialize the classifier specific pipelines
		# Classifier selected by the sub class object in use
        self.initPipeline()
        # End of func return statement
        return
    # End class constructor
    
    # Function to initialize the feature sets
    def initCategories(self, paths):
        self.categories = paths.keys()
        # Loop through the txt files line by line
        # Assign labels to tweets for sentiments in class paths
        for category in paths.keys():
            with open(current_dir + paths[category], "r") as trainingFile:
                for line in trainingFile:
                    self.tweets.append(line)
                    self.labels.append(self.categories.index(category))
        self.labels = np.array(self.labels)
	## The classifiers have to be fitted with two arrays: 
	#   an array X of size [n_samples, n_features] holding the training samples
	#   and an array Y of size [n_samples] holding the target values (class labels) for the training samples
        
	# End of func return statement
        return
    # End initDictSet
    
    ## Function to build classifier pipeline
    ## Default multinomial NB using chi squared statistics
    def initPipeline(self):
        # Pipeline of transformers with a final estimator that behaves like a compound classifier
        self.pipeline = Pipeline([#('vect', CountVectorizer(ngram_range=(1,1))), # Create a vector of feature frequencies
                      ('tfidf', TfidfTransformer(use_idf=False)), # Perform TF-iFD weighting on features
                      #('chi2', SelectKBest(chi2, k=2000)), # Use chi squared statistics to select the k best features
                      ('clf', MultinomialNB())]) # Use the multinomial NB classifier

        # Fit the created multinomial NB classifier
        self.classifier = self.pipeline.fit(self.tweets, self.labels)

        # End func return statement
        return
    # End initPipeline

    # Function to classify input tweet  
    def classify(self, tweet_list):
        # Clean the input list of tweets
        for i in range(0,len(tweet_list)):
            tweet_list[i] = self.cleaner(tweet_list[i])

        # Return predicted class labels for samples in tweet_list
        return self.classifier.predict(tweet_list)
    # End classify func
	## Note: predict_log_proba method, log of probability estimates, is only available for log loss and modified Huber loss
	## This is because when loss="modified_huber", probability estimates may be hard zeros and ones, 
	# 	so taking the logarithm is not possible.
	## It returns the log-probability of the sample for each class in the model
	# 	where classes are ordered as they are in self.classes_.
	## Note: predict_proba, probability estimates, is only available for log loss and modified Huber loss
	## This is because multi class probability estimates are derived from binary (one-versus-all, OVA) estimates 
	# 	by simple normalization, as recommended by Zadrozny and Elkan.
	## It returns the mean accuracy on the given test data and labels

    # Function to get the predicted classifiers confusion matrix
    def getConfusionMatrix(self, actual, predicted):
        print(metrics.classification_report(actual, predicted, target_names=self.categories))
        # Return the confusion matrix
        return metrics.confusion_matrix(actual,predicted)
    # End getConfusionMatrix
    
    ## Function to perform a grid search for best features
    ## GridSearchCV implements a "fit" method and a "predict" method like any classifier 
    #   except that the parameters of the classifier used to predict is optimized by cross-validation.
    def getGridSearch(self):
        # Set the search parameters
        parameters = {'vect__ngram_range': [(1,1),(1,2)], # Try either words or bi grams
                    'vect__max_df': (0.5, 0.1, 0.09),
                    #'vect__max_features': (None, 5000, 10000, 50000),
                    'tfidf__use_idf': (True, False),
                    'tfidf__norm': ('l1', 'l2'),
                    'clf__penalty': ('l2', 'elasticnet', 'l1'), # Default l2
					'clf__alpha': (0.0001, 0.0009), # Default 0.0001
					#'clf_fit_intercept': (True, False), # Default True
                    'clf__n_iter': (5, 50, 25), # Default 1 or 5 depending, optional
					#'clf__random_state':(0, 42), # Default None
                    'clf__epsilon':(0.01, 0.005)} # Default 0.01, depends on classifier (loss)
        # Use all cores to create a grid search
        classifierGS = GridSearchCV(self.pipeline, parameters, n_jobs=-1)
        # Fit the CS estimator for use as a classifier
        classifierGS = classifierGS.fit(self.tweets, self.labels)
        # Get the scores using the GS classifier
        bestParam, score, _ = max(classifierGS.grid_scores_, key=lambda x: x[1])
        # Print the parameter values
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name,bestParam[param_name]))
        # Print the classifier score
        print("Classifier score: " + str(score) + "\n")
        # End of func return statement
        return
    # End getGridSearch 
# End class PosNegClassifier super class

##########################################################################################################################

## Sub class to perform linear support vector machine (SVM) tweet classification
## SGDClassifier arg loss='hinge': (soft-margin) linear Support Vector Machine
## Note: SGDClassifier supports multi class classification by combining multiple 
#	binary classifiers in a "one versus all" (OVA) scheme
class PosNegClassifierLinearSVM(PosNegClassifier):
    # Class constructor
    def __init__(self, paths, cleaner):
        # Call the super class constructor which initializes the classifier
        super(PosNegClassifierLinearSVM, self).__init__(paths, cleaner)
        # End of func return statement
        return
    # End sub class constructor
    
    # Overriding function to build the linear SVM classifier using a pipeline
    def initPipeline(self):
        # Pipeline of transformers with a final estimator that behaves like a compound classifier
        self.pipeline = Pipeline([('vect', CountVectorizer(max_df=0.1, ngram_range=(1,1))), # Create a vector of feature frequencies
                            ('tfidf', TfidfTransformer(norm='l2', use_idf=True)), # Perform TF-iDF weighting on features
                            ('clf', SGDClassifier(alpha=0.0009, epsilon = 0.01, n_iter=50, penalty='elasticnet', random_state=0))]) # Use the SVM classifier
        ## The SGD estimator implements regularized linear models with stochastic gradient descent learning
        ## By default, SGD supports a linear support vector machine (SVM) using the default args below
        ## SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, 
        #   shuffle=True, verbose=0, epsilon=0.1, n_jobs=1, random_state=None, learning_rate='optimal', 
        #   eta0=0.0, power_t=0.5, class_weight=None, warm_start=False, average=False)

        # Fit the created linear SVM classifier
        self.classifier = self.pipeline.fit(self.tweets, self.labels)
        # End of func return statement
        return
    # End initPipeline override
# End PosNegClassifierLinearSVM sub class

##########################################################################################################################

## Sub class to perform quadratic support vector machine (SVM) tweet classification
## SGDClassifier arg loss='squared_hinge' is like hinge, 
#	which is used for linear SVM, but is quadratically penalized.
class PosNegClassifierQuadraticSVM(PosNegClassifier):
    # Class constructor
    def __init__(self, paths, cleaner):
        # Call the super class constructor which initializes the classifier
        super(PosNegClassifierQuadraticSVM, self).__init__(paths, cleaner)
		# End of func return statement
        return
    # End sub class constructor
    
    # Overriding function to build the quadratic SVM classifier using a pipeline
    def initPipeline(self):
        # Pipeline of transformers with a final estimator that behaves like a compound classifier
        self.pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))), # Create a vector of feature frequencies
                            ('tfidf', TfidfTransformer(use_idf=False)), # Perform TF-iDF weighting on features
                            ('clf', SGDClassifier(random_state=42, loss='squared_hinge'))]) # Use the quadratic SVM classifier
        # The SGD estimator implements regularized linear models with stochastic gradient descent learning

        # Fit the created quadratic SVM classifier
        self.classifier = self.pipeline.fit(self.tweets, self.labels)
        # End of func return statement
        return
    # End initPipeline override
# End PosNegClassifierQuadraticSVM sub class

##########################################################################################################################

## Sub class to perform less sensitive support vector machine (SVM) tweet classification
## SGDClassifier arg loss='modified_huber' is another smooth loss that brings tolerance to 
#	outliers as well as probability estimates.
## Note: since they allow to create a probability model, loss="log" 
#	and loss="modified_huber" are more suitable for OVA classification.
class PosNegClassifierModifiedSVM(PosNegClassifier):
    # Class constructor
    def __init__(self, paths, cleaner):
        # Call the super class constructor which initializes the classifier
        super(PosNegClassifierModifiedSVM, self).__init__(paths, cleaner)
        # End of func return statement
        return
    # End sub class constructor
    
    # Overriding function to build the smoothed SVM classifier using a pipeline
    def initPipeline(self):
        # Pipeline of transformers with a final estimator that behaves like a compound classifier
        self.pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), max_df=0.5)), # Create a vector of feature frequencies
                            ('tfidf', TfidfTransformer(use_idf=True, norm='l1')), # Perform TF-iDF weighting on features
                            ('clf', SGDClassifier(random_state=42, loss='modified_huber', penalty='elasticnet', n_iter=50, alpha=1e-05))]) # Use the smoothed SVM classifier
        # The SGD estimator implements regularized linear models with stochastic gradient descent learning

        # Fit the created smoothed SVM classifier
        self.classifier = self.pipeline.fit(self.tweets, self.labels)
        # End of func return statement
        return
    # End initPipeline override
# End PosNegClassifierModifiedSVM sub class

##########################################################################################################################

## Sub class to perform logistic regression tweet classification
## SGDClassifier arg loss='log' performs logistic regression
## Note: since they allow to create a probability model, loss="log" 
#	and loss="modified_huber" are more suitable for OVA classification.
class PosNegClassifierLogSVM(PosNegClassifier):
    # Class constructor
    def __init__(self, paths, cleaner):
        # Call the super class constructor which initializes the classifier
        super(PosNegClassifierLogSVM, self).__init__(paths, cleaner)
        # End of func return statement
        return
    # End sub class constructor
    
    # Overriding function to build the logistic regression classifier using a pipeline
    def initPipeline(self):
        # Pipeline of transformers with a final estimator that behaves like a compound classifier
        self.pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))), # Create a vector of feature frequencies
                            ('tfidf', TfidfTransformer(use_idf=False)), # Perform TF-iDF weighting on features
                            ('clf', SGDClassifier(random_state=42, loss='log'))]) # Use the logistic regression classifier
        # The SGD estimator implements regularized linear models with stochastic gradient descent learning

        # Fit the created logistic regression classifier
        self.classifier = self.pipeline.fit(self.tweets, self.labels)
        # End of func return statement
        return
    # End initPipeline override
# End PosNegClassifierLogSVM sub class
# Note: Using loss="log" or loss="modified_huber" enables the predict_proba method, 
#	which gives a vector of probability estimates per sample.

##########################################################################################################################

## Sub class to perform linear regression tweet classification
## SGDClassifier arg loss='perceptron' is the linear loss used by the perceptron algorithm
## Note: The perceptron algorithm is used for learning weights for features/terms
class PosNegClassifierPerceptronSVM(PosNegClassifier):
    # Class constructor
    def __init__(self, paths, cleaner):
        # Call the super class constructor which initializes the classifier
        super(PosNegClassifierPerceptronSVM, self).__init__(paths, cleaner)
        # End of func return statement
        return
    # End sub class constructor
    
    # Overriding function to build the perceptron algorithm using classifier via a pipeline
    def initPipeline(self):
        # Pipeline of transformers with a final estimator that behaves like a compound classifier
        self.pipeline = Pipeline([('vect', CountVectorizer(max_df=0.5, ngram_range=(1,2))), # Create a vector of feature frequencies
                            ('tfidf', TfidfTransformer(norm='l2', use_idf=True)), # Perform TF-iDF weighting on features
                            ('clf', SGDClassifier(loss='perceptron', alpha=0.0009, epsilon = 0.01, n_iter=25, penalty='l2', random_state=0))]) # Use the perceptron algorithm for classification
		## The SGD estimator implements regularized linear models with stochastic gradient descent learning

        # Fit the created perceptron algorithm using classifier
        self.classifier = self.pipeline.fit(self.tweets, self.labels)
        # End of func return statement
        return
    # End initPipeline override
# End PosNegClassifierPerceptronSVM sub class

##########################################################################################################################

## Sub class to perform linear regression tweet classification
## SGDClassifier arg loss='huber' transforms the squared loss into a linear loss 
# 	over a certain distance, see epsilon arg description in initPipeline func below
## SGDRegressor can also act as a linear SVM using the epsilon_insensitive loss 
# 	function or the slightly different squared_epsilon_insensitive (which penalizes outliers more)
class PosNegClassifierRegression(PosNegClassifier):
    # Class constructor
    def __init__(self, paths, cleaner):
        # Call the super class constructor which initializes the classifier
        super(PosNegClassifierRegression, self).__init__(paths, cleaner)
        # End of func return statement
        return
    # End sub class constructor
    
    # Overriding function to build the linear regression classifier using a pipeline
    def initPipeline(self):
        # Pipeline of transformers with a final estimator that behaves like a compound classifier
        self.pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,2), max_df=0.5)), # Create a vector of feature frequencies
                            ('tfidf', TfidfTransformer(use_idf=True, norm='l2') ), # Perform TF-iDF weighting on features
                            ('clf', SGDClassifier(random_state=42, loss='huber', epsilon=0.001, n_iter=50, alpha=1e-05, penalty='l2'))]) # Use the linear regression classifier
        ## The SGD estimator implements regularized linear models with stochastic gradient descent learning
		## The epsilon arg in the epsilon-insensitive loss functions ('huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive')
		#	For 'huber' it determines the threshold at which it becomes less important to get the prediction exactly right.

        # Fit the created linear regression classifier
        self.classifier = self.pipeline.fit(self.tweets, self.labels)
        # End of func return statement
        return
    # End initPipeline override
# End PosNegClassifierRegression sub class

##########################################################################################################################

## Sub class to perform tweet classification with linear loss
## SGDClassifier arg loss='squred_loss' allows for linear modelling similar to the default SGDRegressor
class PosNegClassifierLossSquared(PosNegClassifier):
    # Class constructor
    def __init__(self, paths, cleaner):
        # Call the super class constructor which initializes the classifier
        super(PosNegClassifierLossSquared, self).__init__(paths, cleaner)
        # End of func return statement
        return
    # End sub class constructor
    
    # Overriding function to build the linear loss classifier using a pipeline
    def initPipeline(self):
        # Pipeline of transformers with a final estimator that behaves like a compound classifier
        self.pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))), # Create a vector of feature frequencies
                            ('tfidf', TfidfTransformer(use_idf=False)), # Perform TF-iDF weighting on features
                            ('clf', SGDClassifier(random_state=42, loss='squared_loss'))]) # Use the classifier for linear loss
        ## The SGD estimator implements regularized linear models with stochastic gradient descent learning
		
        # Fit the created linear loss classifier
        self.classifier = self.pipeline.fit(self.tweets, self.labels)
        # End of func return statement
        return
    # End initPipeline override
# End PosNegClassifierLossSquared sub class

##########################################################################################################################

## Sub class to perform linear regression tweet classification
## SGDRegressor is a linear model fitted by minimizing a regularized empirical loss with SGD
## SGDRegressor mimics a linear regression using the squared_loss loss parameter and it can also act as
# 	a linear SVM using the epsilon_insensitive loss function or the slightly different squared_epsilon_insensitive 
# 	(which penalizes outliers more)
class PosNegRegressor(PosNegClassifier):
    # Class constructor
    def __init__(self, paths, cleaner):
        # Call the super class constructor which initializes the classifier
        super(PosNegRegressor, self).__init__(paths, cleaner)
        # End of func return statement
        return
    # End sub class constructor
    
    # Overriding function to build the regressor using a pipeline
    def initPipeline(self):
        # Pipeline of transformers with a final estimator that behaves like a compound classifier
        self.pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))), # Create a vector of feature frequencies
                            ('tfidf', TfidfTransformer(use_idf=False)), # Perform TF-iDF weighting on features
                            ('clf', SGDRegressor())]) # Use the SGDRegressor classifier
        ## The SGDRegressor estimator works with data represented as dense numpy arrays of floating point values for the features
		## SGDRegressor default mimics linear regression classification
		## SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=5, 
		# 	shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, 
		# 	power_t=0.25, warm_start=False, average=False)

        # Fit the created regressor
        self.classifier = self.pipeline.fit(self.tweets, self.labels)
        # End of func return statement
        return
    # End initPipeline override
# End PosNegRegressor sub class
## Note: SGD stands for Stochastic Gradient Descent, where the gradient of the loss is estimated each sample at a time 
# 	and the model is updated along the way with a decreasing strength schedule (aka learning rate)

##########################################################################################################################

# Sub class for creating a classifier for maximum entropy tweet analysis
class PosNegClassifierMaxEnt(PosNegClassifier):

	# Sub class constructor
    def __init__(self, paths, cleaner):
		# Call the super class constructor
        super(PosNegClassifierMaxEnt, self).__init__(paths, cleaner)
		# End of func return statement
        return
	# End sub class constructor
		
	# Overriding function to build LogisticRegression classifier using a pipeline
    def initPipeline(self):
	    # Pipeline of transformers with a final estimator that behaves like a compound classifier
        self.pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))), # Create a vector of feature frequencies
                            ('tfidf', TfidfTransformer(use_idf=False)), # Perform TF-iDF weighting on features
                            ('clf', LogisticRegression())]) # Use LogisticRegression as the estimator
							
        # Fit the created LogisticRegression classifier
        self.classifier = self.pipeline.fit(self.tweets, self.labels)
		# End of func return statement
        return
	# End initPipeline override
# End PosNegClassifierMaxEnt sub class

##########################################################################################################################

# Sub class for creating a Bernoulli NB classifier for tweet analysis
class PosNegClassifierBNB(PosNegClassifier):

	# Sub class constructor
    def __init__(self, paths, cleaner):
		# Call the super class constructor
        super(PosNegClassifierBNB, self).__init__(paths, cleaner)
		# End of func return statement
        return
	# End sub class constructor
		
	# Overriding function to build BernoulliNB classifier using a pipeline
    def initPipeline(self):
		# Pipeline of transformers with a final estimator that behaves like a compound classifier
        self.pipeline = Pipeline([('vect', CountVectorizer(ngram_range=(1,1))), # Create a vector of feature frequencies
                            ('tfidf', TfidfTransformer(use_idf=False)), # Perform TF-iDF weighting on features
                            ('clf', BernoulliNB())]) # Use the BernoulliNB classifier
							
        # Fit the created BernoulliNB classifier
        self.classifier = self.pipeline.fit(self.tweets, self.labels)
		# End of func return statement
        return
	# End initPipeline override
# End PosNegClassifierBNB sub class

# End script
