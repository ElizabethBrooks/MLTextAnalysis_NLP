import cPickle as cp
import random

name = raw_input("Enter your name (hayden, jeremy, liz): ")
name = name.lower()

otherName = raw_input("Enter the name of the person who you want to test (hayden, jeremy, liz): ")
otherName = otherName.lower()

sents = {'c':'calm', 'e':'excited', 'a':'angry', 'f':'fearful', 's':'sad', 'p':'positive', 'n':'negative', 'u':'neutral', 'o':None}

incorrect = 0

incorrectCounts = {'calm':0, 'excited':0, 'angry':0, 'fearful':0, 'sad':0, 'positive':0, 'negative':0, 'neutral':0}

f = open(otherName + '.pkl', 'rb')
theirTweets = cp.load(f)
f.close()

yourTweets = []

for t in theirTweets:
  yourTweets.append({'tweet': t['tweet'], 'idx':t['idx']})

random.shuffle(yourTweets)

for tweet in yourTweets:
  print tweet['tweet']
  sentiment = ''
  while (not sentiment in sents.keys()):
      sentiment = raw_input("Enter the tweets sentiment: (c-alm, e-xcited, a-angry, f-earful, s-ad, p-ositive, n-egative, u-neutral, or o-ther (not English)): ")
      if sentiment != "":
        sentiment = sentiment.lower()[0]
      else:
        sentiment = ''

  s = sents[sentiment]
  tweet['sentiment'] = s
  s1 = theirTweets[tweet['idx']]['sentiment']
  if s != s1:
    incorrect += 1
    incorrectCounts[s1] += 1

with open(otherName + '_' + name + '.pkl', 'wb') as f:
  cp.dump(yourTweets, f)

print "Percent disagreed: " + str(float(incorrect)/80)
print "Percent disagreed for each sentiment:"
for s in incorrectCounts.keys():
  print s + ": " + str(float(incorrectCounts[s])/10)
