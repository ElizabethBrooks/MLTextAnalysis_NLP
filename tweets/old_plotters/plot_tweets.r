# author: Hayden Fuss

# source for plotting non-numeric data on x-axis, aka using barplot:
# http://craiccomputing.blogspot.com/2011/11/plotting-simple-bar-plot-in-r.html

tweets <- read.csv(file="tweets_per_date_hour.csv", sep=",", head=TRUE)

barplot(tweets[,'number_tweets'], main="Tweets With Keywords vs. Date",
  ylab="Number of Tweets", ylim=c(0,2000), names.arg=tweets[,'date_hour'], las=2)
# barplot: first argument is y-axis, names.arg is labels for x-axis

# look into barchart for "double" bargraphs