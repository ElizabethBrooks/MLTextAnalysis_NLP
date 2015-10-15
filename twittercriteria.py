import yaml
import os
import inspect
import re
import time
import string

# author: Hayden Fuss

# uses os and inspect to determine path to module
myDir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
filepath = myDir + '/twitter_criteria.yml'

# open criteria .yml file and load it into dictionary using yaml
criteria_yml = open(filepath, 'r')
criteria = yaml.load(criteria_yml)
criteria_yml.close()

kw_regex = re.compile('|'.join(criteria['keywords']))

handle_regex = re.compile(r'@[^ :\xe2\"\)\./\\\?\'!@]+')

markup_regex = re.compile('|'.join(criteria['twitterMarkup']))

def getTwitterTimeFmt():
  global criteria
  return criteria['time_fmt']

def getKeywords():
  global criteria
  return criteria['keywords']

def getKeywordRegex():
  global kw_regex
  return kw_regex

def getHandleRegex():
  global handle_regex
  return handle_regex

def getHandlesFromTweet(tweet):
  global handle_regex
  return handle_regex.findall(tweet)

def getTweetDate(tweet_time):
  global criteria
  return time.strptime(tweet_time, criteria['time_fmt'])

def tweetContainsKeyword(tweet):
  global kw_regex
  return kw_regex.search(tweet) is not None

# Function to clean up tweet strings 
# by manually removing irrelevant data (not words)
def cleanUpTweet(tweet_text):
  global markup_regex, handle_regex
  temp = markup_regex.sub(r"", tweet_text)             # removes markup like pictures and &amp;
  temp = re.sub(r"\r|\r\n|\n", r" ", temp)             # replaces newlines with spaces
  temp = unicode(temp, 'utf-8')                        # converts string to utf-8
  temp = temp.encode('unicode_escape')                 # encodes with escape sequences, putting emojis in raw form
  temp = re.sub(r"\\U", r" \\U", temp)                 # puts a space in front of every emoji
  temp = re.sub(r"\\u", r" \\u", temp)                 # puts a space in front of every emoji
  temp = re.sub(r"0fc|\\u201c|\\u201d|\"", r"", temp)  # remove unicode quotes and other markup
  return temp

def has_punc(s):
  for i in range(0, len(s)):
    if s[i] in '!,.;':
      return True
  return

def negation(s):
  s = s.lower()
  return (s == 'not' or 'n\'t' in s or s == 'cant' or s == 'dont' 
    or s == 'cannot')

def cleanForSentiment(tweet_text):
  global markup_regex, handle_regex
  temp = markup_regex.sub(r"", tweet_text)             # removes markup like pictures and &amp;
  temp = re.sub(r"\r|\r\n|\n", r" ", temp)             # replaces newlines with spaces
  temp = handle_regex.sub(r"<HANDLE>", temp)           # replaces @<name> with <HANDLE>
  temp = unicode(temp, 'utf-8')                        # converts string to utf-8
  temp = temp.encode('unicode_escape')                 # encodes with escape sequences, putting emojis in raw form
  temp = re.sub(r"\\U", r" \\U", temp)                 # puts a space in front of every emoji
  temp = re.sub(r"\\u", r" \\u", temp)                 # puts a space in front of every emoji
  temp = re.sub(r"0fc|\\u201c|\\u201d|\"", r"", temp)  # remove unicode quotes and other markup
  res = re.split(' ', temp)                            # splits the string by spaces
  # adds NOT_ in front of every word between a negation and punctuation
  flag = False
  for i in range(0, len(res)):
    if res[i] == 'RT' or '#' in res[i]:
      flag = False
    if flag and (res[i] != '' and res[i][0] != '<' and 
     res[i][:2] != '\U' and res[i][:2] != '\u'):
      res[i] = 'NOT_' + res[i]
    if has_punc(res[i]):
      flag = False
    elif negation(res[i]):
      flag = True
  return ' '.join(res)
