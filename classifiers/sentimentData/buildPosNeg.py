# author: Hayden Fuss

import csv
import numpy as np

sentiments = ['positive', 'negative', 'neutral']

devSize = 300

for s in sentiments:

  test = []

  tweets = []

  with open(s + '_Uncleaned.txt', 'r') as f:
    for line in f:
      tweets.append(line.rstrip('\n'))

  for i in range(0,int(devSize/3.0)):
    j = np.random.randint(len(tweets))
    test.append(tweets.pop(j))

  with open(s + '.txt', 'w') as f:
    f.write('\n'.join(tweets))

  with open(s + 'Test.txt', 'w') as f:
    f.write('\n'.join(test))

