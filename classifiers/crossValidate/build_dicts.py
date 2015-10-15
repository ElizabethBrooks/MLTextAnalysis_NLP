import cPickle as cp
import numpy as np

names = ['hayden', 'liz', 'jeremy']

sentiments = ['angry', 'calm', 'fearful', 'sad', 'excited', 'positive', 'negative', 'neutral']

for n in names:
  tweets = []
  for s in sentiments:
    current = []

    with open(n + '_' + s + '.txt', 'r') as f:
      for line in f:
        current.append(line.rstrip('\n'))

    for i in range(0,10):
      j = np.random.randint(len(current))
      tweets.append({'tweet':current.pop(j), 'sentiment':s, 'idx':len(tweets)})

  with open(n + '.pkl', 'wb') as f:
    cp.dump(tweets, f)