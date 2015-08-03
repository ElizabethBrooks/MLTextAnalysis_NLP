import numpy as np
import csv
import re

# author: Hayden
# source for words: https://github.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/blob/master/data/opinion-lexicon-English/

sentimentWords = []

pos = open('positive.txt')
neg = open('negative.txt')

for word in pos:
    sentimentWords.append(re.escape(word.rstrip('\n')))

for word in neg:
    sentimentWords.append(re.escape(word.rstrip('\n')))

pos.close()
neg.close()

randIndices = np.random.randint(low=0, high=len(sentimentWords), size=200)

words = []

for i in randIndices:
  words.append(sentimentWords[i])

del(sentimentWords)

regex = re.compile('|'.join(words))

del(words)

tweets = []

with open('geo_02.csv') as csvfile:
  tweetData = csv.DictReader(csvfile)
  for datum in tweetData:
    if datum['tweet_text'] != "" and regex.search(datum['tweet_text'].lower()) is not None:
      tweets.append(datum['tweet_text'])


randIndices = np.random.randint(low=0, high=len(tweets), size=1000)

randomTweets = []

for i in randIndices:
  if not tweets[i] in randomTweets:
    randomTweets.append(tweets[i])

while len(randomTweets) < 1000:
  randIndices = np.random.randint(low=0, high=len(tweets), size=1000)
  for i in randIndices:
    if not tweets[i] in randomTweets:
      randomTweets.append(tweets[i])
    if len(randomTweets) >= 1000:
      break

for tweet in randomTweets:
  print tweet
