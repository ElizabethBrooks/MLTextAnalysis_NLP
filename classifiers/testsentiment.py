# Author: Hayden Fuss
# File: testsentiment.py
# Date Modified: 07/31/2015
# Edited: Elizabeth Brooks

import tweetclassifier as tc
import os
import numpy as np

currentDir = os.getcwd()

def cleanNothing(tweet):
  return tweet

trainingPaths = {'angry':'/sentimentData/angryTraining.txt', 'sad':'/sentimentData/sadTraining.txt',
 'calm':'/sentimentData/calmTraining.txt', 'fearful':'/sentimentData/fearfulTraining.txt', 'positive':'/sentimentData/positiveTraining.txt',
 'excited':'/sentimentData/excitedTraining.txt', 'negative':'/sentimentData/negativeTraining.txt', 'neutral':'/sentimentData/neutralTraining.txt'}

categories = trainingPaths.keys()

testPaths = {'angry':'/sentimentData/angryValidation.txt', 'sad':'/sentimentData/sadValidation.txt',
 'calm':'/sentimentData/calmValidation.txt', 'fearful':'/sentimentData/fearfulValidation.txt', 'positive':'/sentimentData/positiveValidation.txt',
 'excited':'/sentimentData/excitedValidation.txt', 'negative':'/sentimentData/negativeValidation.txt', 'neutral':'/sentimentData/neutralValidation.txt'}

actual = np.array([])

testTweets = []

for p in testPaths.keys():
  temp = []
  with open(currentDir + testPaths[p], 'r') as f:
    for line in f:
      temp.append(line.rstrip('\n'))
  testTweets.extend(temp)
  temp = np.empty(len(temp))
  temp.fill(categories.index(p))
  actual = np.append(actual, temp)

def testClassifier(clssfr):
  print "Testing " + type(clssfr).__name__ + "..."
  predicted = clssfr.classify(testTweets)
  mat = clssfr.getConfusionMatrix(actual, predicted)
  num = len(mat)
  correct = 0
  total = 0
  for i in range(0, num):
    for j in range(0, num):
      total += mat[i][j]
      if i == j:
        correct += mat[i][j]
  #print correct
  #print total
  print "Accuracy: " + str(float(correct)/total) 
  print mat
  print "...done.\n"

relClssr = tc.TweetClassifier(trainingPaths, cleanNothing)
testClassifier(relClssr)
relClssr.getGridSearch()

relClssr = tc.TweetClassifierLinearSVM(trainingPaths, cleanNothing)
testClassifier(relClssr)
relClssr.getGridSearch()

relClssr = tc.TweetClassifierModifiedSVM(trainingPaths, cleanNothing)
testClassifier(relClssr)
relClssr.getGridSearch()

relClssr = tc.TweetClassifierQuadraticSVM(trainingPaths, cleanNothing)
testClassifier(relClssr)
relClssr.getGridSearch()

relClssr = tc.TweetClassifierLogSVM(trainingPaths, cleanNothing)
testClassifier(relClssr)
relClssr.getGridSearch()

relClssr = tc.TweetClassifierPerceptronSVM(trainingPaths, cleanNothing)
testClassifier(relClssr)
relClssr.getGridSearch()

relClssr = tc.TweetClassifierLossSquared(trainingPaths, cleanNothing)
testClassifier(relClssr)
relClssr.getGridSearch()

relClssr = tc.TweetClassifierMaxEnt(trainingPaths, cleanNothing)
testClassifier(relClssr)
relClssr.getGridSearch()

relClssr = tc.TweetClassifierBNB(trainingPaths, cleanNothing)
testClassifier(relClssr)
relClssr.getGridSearch()

relClssr = tc.TweetClassifierRegression(trainingPaths, cleanNothing)
testClassifier(relClssr)
relClssr.getGridSearch()
