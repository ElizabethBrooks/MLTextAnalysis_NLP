# author: Hayden Fuss

import csv
import numpy as np
import re

unknownTweets = []

fields = None

with open('cleaned_geo_tweets_4_12_22.csv') as csvfile:
  tweets = csv.DictReader(csvfile)
  fields = tweets.fieldnames
  for t in tweets:
    unknownTweets.append(t)

print len(unknownTweets)

trainingTweets = []

for i in range(0,3600):
  j = np.random.randint(len(unknownTweets))
  trainingTweets.append(unknownTweets.pop(j))

print len(trainingTweets)
print len(unknownTweets)

with open('test_tweets_4_12_22.csv', 'wb') as csvfile:
  tweets = csv.DictWriter(csvfile, fieldnames=fields)
  tweets.writeheader()
  for t in unknownTweets:
    tweets.writerow(t)

trainingText = []

with open('training_tweets_4_12_22.csv', 'wb') as csvfile:
  tweets = csv.DictWriter(csvfile, fieldnames=fields)
  tweets.writeheader()
  for t in trainingTweets:
    trainingText.append(re.sub(r"\r|\r\n|\n", r" ", t['tweet_text']))
    tweets.writerow(t)

print len(trainingText)
del unknownTweets
del trainingTweets

trainingTweets = {'hayden':{'1':trainingText[:400], '2':trainingText[1200:1600], '3':trainingText[2400:2800]},
                  'liz':{'1':trainingText[400:800], '2':trainingText[1600:2000], '3':trainingText[2800:3200]},
                  'jeremy':{'1':trainingText[800:1200], '2':trainingText[2000:2400], '3':trainingText[3200:3600]}}

for name in trainingTweets.keys():
  for num in trainingTweets[name].keys():
    with open(name + '_' + num + '.txt', 'w') as f:
      f.write('\n'.join(trainingTweets[name][num]))




