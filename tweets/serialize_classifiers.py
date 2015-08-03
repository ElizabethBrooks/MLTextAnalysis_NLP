import cPickle as cp

import os, sys
sys.path.append(os.path.realpath('./classifiers/'))
import tweetclassifier as tc

import twittercriteria as twc

paths = {'sad':'/sadTraining.txt', 'fearful':'/fearfulTraining.txt', 'angry':'/angryTraining.txt', 'calm':'/calmTraining.txt',
 'excited':'/excitedTraining.txt', 'other':'/otherTraining.txt'}

objs = {'mnb':tc.TweetClassifier, 'linSVM':tc.TweetClassifierLinearSVM, 'quadSVM':tc.TweetClassifierQuadraticSVM,
  'modSVM':tc.TweetClassifierModifiedSVM, 'logSVM':tc.TweetClassifierLogSVM, 'percSVM':tc.TweetClassifierPerceptronSVM,
  'regrSVM':tc.TweetClassifierRegression, 'lsqrdSVM':tc.TweetClassifierLossSquared, 'regressor':tc.TweetRegressor, 
  'maxent':tc.TweetClassifierMaxEnt, 'bnb':tc.TweetClassifierBNB}

clssfr = None

for name in objs.keys():
  clssfr = objs[name](paths, twc.cleanForSentiment)
  with open(name + '.pkl', 'wb') as f:
    cp.dump(clssfr, f)
  del clssfr
  print "Done with " + name 