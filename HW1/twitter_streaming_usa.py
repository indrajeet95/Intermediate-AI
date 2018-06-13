import sys
import os
import jsonpickle
import tweepy

access_token = "76254882-igW2dzDAFfazDiUHFCZZRdWVRqP7l3L3aFSURPL9q"
access_token_secret = "SuW1MfXZtBB1JJ7iXcSuUA4jxlneMAQdPQzWKVo22NpS8"
consumer_key = "N2K1qgJDKqsYkvqNXEa1eNjOV"
consumer_secret = "Cx0bzYKvX8an4v2hPl4Sc1xnXVnLI8Hv7heg9IvscS2c82242a"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

if (not api):
    print ("Problem connecting to API")

places = api.geo_search(query="USA", granularity="country")

place_id = places[0].id
print('USA id is: ',place_id)

auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)
api = tweepy.API(auth, wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

if (not api):
    print ("Problem Connecting to API")

searchQuery = 'place:96683cc9126741d1'
maxTweets = 1000000
tweetsPerQry = 100
tweetCount = 0

with open('USA_Tweets.json', 'w') as f:
    for tweet in tweepy.Cursor(api.search,q=searchQuery).items(maxTweets) :
        if tweet.place is not None:
            f.write(jsonpickle.encode(tweet.text, unpicklable=False) + '\n')
            tweetCount += 1
    print("Downloaded {0} tweets".format(tweetCount))
