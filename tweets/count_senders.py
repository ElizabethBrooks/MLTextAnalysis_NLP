import csv

# author: Josh Lojzim

# source: https://docs.python.org/2/library/csv.html
# uses the csv.DictReader class to parse the data
# because the first line of the file explains each field in a csv
# format, DictReader immediately knows how to parse each line into
# a hash/dictionary with the keys matching the fields specified in 
# the first line of the file

tweets_sender_id = {}

with open('cleaned_geo_tweets_Apr_12_to_22.csv') as csvfile:
  tweets = csv.DictReader(csvfile)
  for tweet in tweets:
    if not tweet['sender_id'] in tweets_sender_id.keys():
      tweets_sender_id[tweet['sender_id']] = 1
    else:
      tweets_sender_id[tweet['sender_id']] = tweets_sender_id[tweet['sender_id']] + 1

print "sender_id,count"
for t in sorted(tweets_sender_id):
  print t + ","+ str(tweets_sender_id[t])  
print "\n"
