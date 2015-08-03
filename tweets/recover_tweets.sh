#!/bin/bash

# author: Hayden Fuss

for i in 02 03 04 05 06 07 08 09; do

  cd $i

  pwd

  rm -rf *.csv *.sql

  dumpname="geo_tweets_2013_04_${i}.dump"

  csvname="geo_${i}.csv"

  # restores database from pgdump file
  pg_restore $dumpname > geo.sql

  # replaces all tabs with commas, making it csv format
  sed 's/	/,/g' < geo.sql > $csvname

  rm -rf geo.sql

  # removes first 35 lines and last 22 lines
  < $csvname tail -n +35 | tail -r | tail -n +23 | tail -r > temp.csv

  # adds csv header to first line
  sed '1i\
  tweet_id,time,lat,lon,goog_x,goog_y,sender_id,sender_name,source,reply_to_user_id,reply_to_tweet_id,place_id,tweet_text\
  ' temp.csv > $csvname

  rm -rf temp.csv

  ls *.csv

  cd ..

done
