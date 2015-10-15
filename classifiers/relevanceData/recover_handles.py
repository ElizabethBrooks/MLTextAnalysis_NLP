import sys, os
sys.path.append(os.path.realpath('../../'))
import twittercriteria as twc

name = raw_input("Enter name (hayden, jeremy, liz): ")
name = name.lower()

oldRel = []
with open(name + '_1_relevant.txt', 'r') as f:
  for line in f:
    oldRel.append(line.rstrip('\n'))

oldIrr = []
with open(name + '_1_irrelevant.txt', 'r') as f:
  for line in f:
    oldIrr.append(line.rstrip('\n'))

newRel = []

newIrr = []

with open(name + '_1.txt', 'r') as f:
  for line in f:
    tweet = line.rstrip('\n')
    cleaned = twc.cleanForSentiment(tweet)
    if cleaned in oldRel:
      newRel.append(twc.cleanUpTweet(tweet))
    elif cleaned in oldIrr:
      newIrr.append(twc.cleanUpTweet(tweet))

with open(name + '_recleaned_relevant.txt', 'w') as f:
  f.write('\n'.join(newRel))

with open(name + '_recleaned_irrelevant.txt', 'w') as f:
  f.write('\n'.join(newIrr))