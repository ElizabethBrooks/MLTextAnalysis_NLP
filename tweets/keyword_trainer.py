# author: Hayden Fuss

import csv
import twittercriteria as twc
import time
import numpy as np

tweet_time_fmt = twc.getTwitterTimeFmt()

def randomSubset(alist, sub_size=500):
  temp = []
  indices = np.random.randint(low=0, high=len(alist), size=sub_size)
  for i in indices:
    temp.append(alist[i])
  return temp

trueIrrelevants = []

possibleRelevants = []

with open('cleaned_geo_tweets_Apr_12_to_22.csv') as csvfile:
  tweetData = csv.DictReader(csvfile)
  for tweet in tweetData:
    if tweet['time'] != "":
      # parse date/time into object
      date = time.strptime(tweet['time'], tweet_time_fmt)
      tweet['tweet_text'] = twc.cleanUpTweet(tweet['tweet_text'])
      if date.tm_mday < 15:
        trueIrrelevants.append(tweet['tweet_text'])
      elif twc.tweetContainsKeyword(tweet['tweet_text']):
        possibleRelevants.append(tweet['tweet_text'])

trueIrrelevants = randomSubset(trueIrrelevants)
possibleRelevants = randomSubset(possibleRelevants)

trueRelevants = []

for each in possibleRelevants:
  print each
  result = raw_input("Enter a r for relevant, i for irrelevant, n for neither (not English): ")
  result = result.lower()
  if result != '':
    if result[0] == 'i':
      trueIrrelevants.append(each)
    elif result[0] == 'r':
      trueRelevants.append(each)

with open('irrelevantTraining.txt', 'w') as ir:
  for i in trueIrrelevants:
    ir.write(i + '\n')

with open('relevantTraining.txt', 'w') as rr:
  for r in trueRelevants:
    rr.write(r + '\n')


