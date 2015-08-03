# Author: Hayden Fuss
# File: testrelevance.py
# Date Modified: 07/31/2015
# Edited: Elizabeth Brooks

import relevanceclassifier as tc
import os
import numpy as np

currentDir = os.getcwd()

def cleanNothing(tweet):
  return tweet

trainingPaths = {'relevent':'/relevanceData/relevantTraining.txt', 'irrelevent':'/relevanceData/irrelevantTraining.txt'}

categories = trainingPaths.keys()

testPaths = {'relevent':'/relevanceData/relevantTest.txt', 'irrelevent':'/relevanceData/irrelevantTest.txt'}

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

# The main method used for testing classifiers
def main():  
	relClssr = tc.RelevanceClassifier(trainingPaths, cleanNothing)
	testClassifier(relClssr)
	#relClssr.getGridSearch()

	relClssr = tc.RelevanceClassifierLinearSVM(trainingPaths, cleanNothing)
	testClassifier(relClssr)
	#relClssr.getGridSearch()

	relClssr = tc.RelevanceClassifierModifiedSVM(trainingPaths, cleanNothing)
	testClassifier(relClssr)
	#relClssr.getGridSearch()

	relClssr = tc.RelevanceClassifierQuadraticSVM(trainingPaths, cleanNothing)
	testClassifier(relClssr)
	#relClssr.getGridSearch()

	relClssr = tc.RelevanceClassifierLogSVM(trainingPaths, cleanNothing)
	testClassifier(relClssr)
	#relClssr.getGridSearch()

	relClssr = tc.RelevanceClassifierPerceptronSVM(trainingPaths, cleanNothing)
	testClassifier(relClssr)
	#relClssr.getGridSearch()

	relClssr = tc.RelevanceClassifierLossSquared(trainingPaths, cleanNothing)
	testClassifier(relClssr)
	#relClssr.getGridSearch()

	relClssr = tc.RelevanceClassifierMaxEnt(trainingPaths, cleanNothing)
	testClassifier(relClssr)
	#relClssr.getGridSearch()

	relClssr = tc.RelevanceClassifierBNB(trainingPaths, cleanNothing)
	testClassifier(relClssr)
	#relClssr.getGridSearch()

	relClssr = tc.RelevanceClassifierRegression(trainingPaths, cleanNothing)
	testClassifier(relClssr)
	#relClssr.getGridSearch()

#Necessary for Windows OS
if __name__ == '__main__':
	main()