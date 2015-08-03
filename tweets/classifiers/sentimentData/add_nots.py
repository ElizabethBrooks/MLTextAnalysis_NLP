import os
import sys
sys.path.append(os.path.realpath('../../'))

import twittercriteria as twc

sentiments = ['angry', 'fearful', 'calm', 'excited', 'sad', 'positive', 'negative', 'neutral']

for s in sentiments:
  tweets = []
  with open(s + 'Training.txt') as infile:
    for line in infile:
      line.rstrip('\n')
      tweets.append(twc.cleanForSentiment(line))
  with open(s + 'Not.txt', 'w') as out:
    out.write('\n'.join(tweets))