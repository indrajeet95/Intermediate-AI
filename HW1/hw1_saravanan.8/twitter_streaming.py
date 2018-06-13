import sys
try:
    import json
except ImportError:
    import simplejson as json

try:
    from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
except:
    print "Cannot find Python Twitter Tools."
    sys.exit()

ACCESS_TOKEN = "76254882-igW2dzDAFfazDiUHFCZZRdWVRqP7l3L3aFSURPL9q"
ACCESS_SECRET = "SuW1MfXZtBB1JJ7iXcSuUA4jxlneMAQdPQzWKVo22NpS8"
CONSUMER_KEY = "N2K1qgJDKqsYkvqNXEa1eNjOV"
CONSUMER_SECRET = "Cx0bzYKvX8an4v2hPl4Sc1xnXVnLI8Hv7heg9IvscS2c82242a"

class TwitterStreaming:
    general_tweets = "general_tweets.json"
    usa_tweets = "usa_tweets.json"
    usa_coordinates = "-125.85,31.35,-62.75,48.34"

    def fetch_stream_to_file(self, tweet_count, filename):
        output_file = open(filename, "w+")
        oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
        twitter_stream = TwitterStream(auth=oauth)
        iterator = twitter_stream.statuses.sample()
        for tweet in iterator:
            if 'text' not in tweet:
                continue
            tweet_count -= 1
            output_file.write(json.dumps(tweet) + "\n")
            if tweet_count <= 0:
                break
        output_file.close()

    def fetch_local_tweet_stream(self, locations, tweet_count, filename):
        output_file = open(filename, "a")
        oauth = OAuth(ACCESS_TOKEN, ACCESS_SECRET, CONSUMER_KEY, CONSUMER_SECRET)
        twitter_stream = TwitterStream(auth=oauth)
        iterator = twitter_stream.statuses.filter(locations=locations)
        for tweet in iterator:
            if 'text' not in tweet:
                continue
            tweet_count -= 1
            output_file.write(json.dumps(tweet) + "\n")
            if tweet_count <= 0:
                break
        output_file.close()

    def run_main(self):
        print "Random Twitter stream\n"
        self.fetch_stream_to_file(15000, self.general_tweets)
        print "USA Twitter stream\n"
        self.fetch_local_tweet_stream(self.usa_coordinates, 1000, self.usa_tweets)

solution = TwitterStreaming()
solution.run_main()
