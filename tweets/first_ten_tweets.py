#!/usr/bin/python2.7

# author: Hayden Fuss

"""
  Prints the first 10 tweets as raw data.
"""

import csv

# source for printing first n rows: 
# http://stackoverflow.com/questions/7661540/print-the-first-two-rows-of-a-csv-file-to-a-standard-output

n = 10

with open('cleaned_geo_tweets_Apr_12_to_22.csv') as csvfile:
  tweets = csv.DictReader(csvfile)
  for i in range(n):
    print "Tweet #" + str(i)
    tweetData = tweets.next() # gets the next hash from the reader
    print str(tweetData) + "\n"
