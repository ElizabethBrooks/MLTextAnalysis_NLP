# author: Hayden Fuss

import numpy as np

sentiments = {'angry':0.098, 'calm':0.0768, 'fearful':0.037, 'sad':0.081, 'excited':0.147, 'positive':0.095, 'negative':0.123, 'neutral':0.342}

devSize = 100

for s in sentiments.keys():

  test = []

  tweets = []

  with open('liz_' + s + '.txt', 'r') as f:
    for line in f:
      tweets.append(line.rstrip('\n'))

  for i in range(0,int(sentiments[s]*devSize)):
    j = np.random.randint(len(tweets))
    test.append(tweets.pop(j))

  with open(s + 'Training.txt', 'w') as f:
    f.write('\n'.join(tweets))

  with open(s + 'Validation.txt', 'w') as f:
    f.write('\n'.join(test))

