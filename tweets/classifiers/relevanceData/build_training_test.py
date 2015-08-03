# author: Hayden Fuss

import csv
import numpy as np

relTest = []

relTweets = []

with open('relevant.txt', 'r') as f:
  for line in f:
    relTweets.append(line.rstrip('\n'))

for i in range(0,10):
  j = np.random.randint(len(relTweets))
  relTest.append(relTweets.pop(j))


irrelTest = []

irrelTweets = []

with open('irrelevant.txt', 'r') as f:
  for line in f:
    irrelTweets.append(line.rstrip('\n'))

for i in range(0,90):
  j = np.random.randint(len(irrelTweets))
  irrelTest.append(irrelTweets.pop(j))

with open('relevantTraining.txt', 'w') as f:
  f.write('\n'.join(relTweets))

with open('relevantTest.txt', 'w') as f:
  f.write('\n'.join(relTest))

with open('irrelevantTraining.txt', 'w') as f:
  f.write('\n'.join(irrelTweets))

with open('irrelevantTest.txt', 'w') as f:
  f.write('\n'.join(irrelTest))

