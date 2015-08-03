# author: Hayden Fuss

senders <- read.csv(file="sender_count.csv", sep=",", head=TRUE)

sendersMean <- mean(senders$count)
sendersSD <- sd(senders$count)
sendersMed <- median(senders$count)

hist(rep(seq(1, 15872, 1), senders$count), breaks=0:15872, main="Number of Tweets per Sender", 
  xlab="Senders", ylab="Number of Tweets")

text(x=5000, y=1500, labels=paste("Mean =", sendersMean, "\nMedian =", 
  sendersMed, "\nStd. Dev =", sendersSD) )