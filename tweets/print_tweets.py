#!/usr/bin/python2.7

# author: Hayden Fuss

"""
  Prints the time, lon/lat, and text of all the tweets in the given dataset
  while determining number of Boston Strong hashtags.
"""

import csv

# source: https://docs.python.org/2/library/csv.html
# uses the csv.DictReader class to parse the data
# because the first line of the file explains each field in a csv
# format, DictReader immediately knows how to parse each line into
# a hash/dictionary with the keys matching the fields specified in 
# the first line of the file

numBostonStrongTags = 0

with open('cleaned_geo_tweets_Apr_12_to_22.csv') as csvfile:
  tweets = csv.DictReader(csvfile)
  i = 0
  # for all the tweets the reader finds
  for tweetData in tweets:
    print "Tweet #" + str(i)
    i = i + 1
    print tweetData['time']
    print tweetData['lat'] + ", " + tweetData['lon']
    print tweetData['tweet_text'] + "\n"
    tweetData['tweet_text'] = tweetData['tweet_text'].lower()
    if "#bostonstrong" in tweetData['tweet_text']: numBostonStrongTags = numBostonStrongTags + 1

print numBostonStrongTags
