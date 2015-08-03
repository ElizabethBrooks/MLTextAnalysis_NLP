
# ## Useful libraries

import time                         # time/date parser
import csv                          # data parser
import numpy as np                  # arrays for plotting
import matplotlib.pyplot as plt     # plotting
import math                         # ceiling for y-max in plots
import twittercriteria as twc       # yaml, re, os

import os
import sys
sys.path.append(os.path.realpath('../census/'))
import bostonmap as bm
# sys.path.append(os.path.realpath('./classifiers/'))
# import tweetclassifier as tc
import re

# paths = {'sad':'/sadTraining.txt', 'fearful':'/fearfulTraining.txt', 'angry':'/angryTraining.txt', 'calm':'/calmTraining.txt',
#  'excited':'/excitedTraining.txt', 'other':'/otherTraining.txt'}

# cats = paths.keys()

# clssfr = tc.TweetClassifierLinearSVM(paths, twc.cleanForSentiment)

handles = ['@boston_police',
           '@bostonglobe',
           '@bostonmarathon',
           '@redsox',
           '@barackobama',
           '@bostondotcom',
           '@stoolpresidente',
           '@nhlbruins',
           '@mlb',
           '@cnnbrk',
           '@ap',
           '@7news',
           '@fox25news',
           '@youranonnews',
           '@cnn',
           '@middlebrooks',
           '@dzhokhar_a',
           '@wcvb',
           '@cbsboston',
           '@universalhub',
           '@tdgarden',
           '@massstatepolice',
           '@j_tsar',
           '@huffingtonpost',
           '@wbcsays']

handle_reg = re.compile('|'.join(handles))

def containsHandle(tweet):
    return handle_reg.search(tweet) is not None

def getTimeString(currentHour):
    timeStr = ""
    if currentHour != 12 and currentHour != 24:
        timeStr += str(currentHour%12)
    else:
        timeStr += str(12)
    if currentHour < 12 or currentHour == 24:
        timeStr += ' AM'
    else:
        timeStr += ' PM'
    return timeStr

# ## Retrieving useful data with our `twitter_criteria` module
# retrieve the time format string from the criteria dictionary in twc
tweet_time_fmt = twc.getTwitterTimeFmt()

### get data

kwTweets = []
currentHour = 14

infoTweets = []

# sentimentTweets = {}
# for c in cats:
#     sentimentTweets[c] = []

# tweetList = []
# textList = []

with open('cleaned_geo_tweets_4_12_22.csv') as csvfile:
    # reads first line of csv to determine keys for the tweet hash, tweets 
    # is an iterator through the list of tweet hashes the DictReader makes
    tweets = csv.DictReader(csvfile)
    # for all the tweets the reader finds
    for tweetData in tweets:
        # make sure its not a 'false tweet' from people using newlines in their tweet_text's
        if tweetData['time'] != "":
            # parse date/time into object
            date = time.strptime(tweetData['time'], tweet_time_fmt)
            if date.tm_mday == 15 and twc.tweetContainsKeyword(tweetData['tweet_text']):
            #if date.tm_mday == 15:
                if date.tm_hour == currentHour:
                    kwTweets.append(tweetData)
                    #tweetList.append(tweetData)
                    #textList.append(tweetData['tweet_text'])
                    if containsHandle(tweetData['tweet_text']):
                        infoTweets.append(tweetData)
                elif date.tm_hour == currentHour + 1:
                    currentHour += 1
                    timeStr = getTimeString(currentHour)
                    # results = clssfr.classify(textList)
                    # for i in range(0, len(results)):
                    #     sentimentTweets[cats[results[i]]].append(tweetList[i])

                    # for sentiment in sentimentTweets.keys():
                    #     boston = bm.GreaterBostonScatter(sentimentTweets[sentiment])
                    #     boston.plotMap(outname=sentiment + '_bombing_scatter_' + str(currentHour),
                    #         title='Locations of ' + sentiment.title() + ' Tweets At ' + timeStr)
                    # boston = bm.GreaterBostonDensity(kwTweets)
                    # boston.plotMap(outname='bombingDay_density_'+str(currentHour), 
                    #     title='Density of Keyword Tweets At ' + timeStr)
                    boston = bm.GreaterBostonScatter(kwTweets)
                    boston.plotMap(outname='bombingDay_keyword_'+str(currentHour),
                        title='Locations of Keyword Tweets At ' + timeStr)

                    boston = bm.GreaterBostonScatter(infoTweets)
                    boston.plotMap(outname='bombingDay_kw_info_'+str(currentHour),
                        title='Locations of Keyword Tweets At ' + timeStr)

                    kwTweets.append(tweetData)
                    if containsHandle(tweetData['tweet_text']):
                        infoTweets.append(tweetData)
                    plt.close('all')
                    print "Done with " + timeStr
                    # del tweetList
                    # del textList
                    # tweetList = [tweetData]
                    # textList = [tweetData['tweet_text']]


currentHour += 1
timeStr = getTimeString(currentHour)
# results = clssfr.classify(textList)
# for i in range(0, len(results)):
#     sentimentTweets[cats[results[i]]].append(tweetList[i])

# for sentiment in sentimentTweets.keys():
#     boston = bm.GreaterBostonScatter(sentimentTweets[sentiment])
#     boston.plotMap(outname=sentiment + '_bombing_scatter_' + str(currentHour),
#         title='Locations of ' + sentiment.title() + ' Tweets At ' + timeStr)

#     boston = bm.GreaterBostonDensity(sentimentTweets[sentiment])
#     boston.plotMap(outname=sentiment + '_bombing_density_' + str(currentHour),
#         title='Density of ' + sentiment.title() + ' Tweets At ' + timeStr)
boston = bm.GreaterBostonScatter(kwTweets)
boston.plotMap(outname='bombingDay_scatter_'+str(currentHour),
    title='Locations of Keyword Tweets At ' + timeStr)

boston = bm.GreaterBostonScatter(infoTweets)
boston.plotMap(outname='bombingDay_kw_info_'+str(currentHour),
    title='Locations of Keyword Tweets At ' + timeStr)

# boston = bm.GreaterBostonDensity(kwTweets)
# del(kwTweets)
# boston.plotMap(outname='bombingDay_density_'+str(currentHour), 
#     title='Density of Keyword Tweets At ' + timeStr)
plt.close('all')
print "Done with " + timeStr
