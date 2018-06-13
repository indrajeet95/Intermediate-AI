# Import the necessary package to process data in JSON format
try:
    import json
except ImportError:
    import simplejson as json

import pandas as pd
import matplotlib.pyplot as plt

a = 0
b = 0
# We use the file saved from last step as example
tweets_filename = 'twitter_stream_1000tweets.txt'
tweets_file = open(tweets_filename, "r")

for line in tweets_file:
    try:
        # Read in one line of the file, convert it into a json object
        tweet = json.loads(line.strip())

        if 'text' in tweet: # only messages contains 'text' field is a tweet
            if tweet['id']:
                print tweet['id'] # This is the tweet's id
                a = a + 1
            # print tweet['created_at'] # when the tweet posted
            # print tweet['text'] # content of the tweet
            if tweet['lang']:
                print tweet['lang']
                b = b + 1
            # print tweet['user']['id'] # id of the user who posted the tweet
            # print tweet['user']['name'] # name of the user, e.g. "Wei Xu"
            # print tweet['user']['screen_name'] # name of the user account, e.g. "cocoweixu"

            # hashtags = []
            # for hashtag in tweet['entities']['hashtags']:
            	# hashtags.append(hashtag['text'])
            # print hashtags

    except:
        # read in a line is not in JSON format (sometimes error occured)
        continue

print 'No of Tweets: ' + str(a)
print ' No of Tweets that have Lang ID: ' + str(b)
# http://adilmoujahid.com/posts/2014/07/twitter-analytics/
