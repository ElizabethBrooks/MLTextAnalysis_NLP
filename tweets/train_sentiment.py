import twittercriteria as twc

name = raw_input("Enter your name (hayden, jeremy, liz): ")
name = name.lower()

num = 0
while (num != 1 and num != 2 and num != 3):
  num = raw_input("Enter which set you want to do (1,2, or 3): ")
  num = int(num)

fearful = []
excited = []
angry = []
sad = []
positive = []
calm = []
negative = []
neutral = []

relevant = []
irrelevant = []

sents = ['c', 'e', 'a', 'f', 's', 'p', 'n', 'u', 'o']
rels = ['r', 'i', 'n']

with open(name + "_" + str(num) + ".txt", "r") as inFile:
  for tweet in inFile:
    tweet = tweet.rstrip('\n')
    print tweet
    tweet = twc.cleanForSentiment(tweet)
    sentiment = ''
    while (not sentiment in sents):
        sentiment = raw_input("Enter the tweets sentiment: (c-alm, e-xcited, a-angry, f-earful, s-ad, p-ositive, n-egative, u-neutral, or o-ther (not English)): ")
        if sentiment != "":
          sentiment = sentiment.lower()[0]
        else:
          sentiment = ''

    if sentiment == 'c':
      calm.append(tweet)
    elif sentiment == 'a':
      angry.append(tweet)
    elif sentiment == 'f':
      fearful.append(tweet)
    elif sentiment == 's':
      sad.append(tweet)
    elif sentiment == 'p':
      positive.append(tweet)
    elif sentiment =='n':
      negative.append(tweet)
    elif sentiment == 'e':
      excited.append(tweet)
    elif sentiment == 'u':
      neutral.append(tweet)

    if num == 1:
      relevance = ''
      while (not relevance in rels):
          relevance = raw_input("Enter the tweets relevance: (r-elevant, i-rrelevant or n-one (not English)): ")
          if relevance != "":
            relevance = relevance.lower()[0]
          else:
            relevance = ''

      if relevance == 'r':
        relevant.append(tweet)
      elif relevance == 'i':
        irrelevant.append(tweet)


if num == 1:
  with open(name + "_" + str(num) + "_relevant.txt", "w") as out:
    out.write('\n'.join(relevant))

  with open(name + "_" + str(num) + "_irrelevant.txt", "w") as out:
    out.write('\n'.join(irrelevant))


with open(name + "_" + str(num) + "_fearful.txt", "w") as out:
  out.write('\n'.join(fearful))


with open(name + "_" + str(num) + "_calm.txt", "w") as out:
  out.write('\n'.join(calm))


with open(name + "_" + str(num) + "_angry.txt", "w") as out:
  out.write('\n'.join(angry))


with open(name + "_" + str(num) + "_excited.txt", "w") as out:
  out.write('\n'.join(excited))


with open(name + "_" + str(num) + "_sad.txt", "w") as out:
  out.write('\n'.join(sad))


with open(name + "_" + str(num) + "_positive.txt", "w") as out:
  out.write('\n'.join(positive))


with open(name + "_" + str(num) + "_negative.txt", "w") as out:
  out.write('\n'.join(negative))

with open(name + "_" + str(num) + "_neutral.txt", "w") as out:
  out.write('\n'.join(neutral))
