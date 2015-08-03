import csv
import twittercriteria as twc

counts = {}

with open('cleaned_geo_tweets_4_12_22.csv') as csvfile:
  tweets = csv.DictReader(csvfile)
  for t in tweets:
    date = twc.getTweetDate(t['time'])
    if not date.tm_mday in counts:
      counts[date.tm_mday] = {'i':0, 'we':0, 'our':0, 'my':0}
    tokens = t['tweet_text'].split()
    for tok in tokens:
      tok = tok.lower()
      if tok in counts[date.tm_mday].keys():
        counts[date.tm_mday][tok] += 1

for d in sorted(counts):
  print "4/" + str(d)
  for word in counts[d].keys():
    print "\t" + word.title() + ": " + str(counts[d][word])
  print ""
