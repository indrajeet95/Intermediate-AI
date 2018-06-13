from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream

access_token = "76254882-igW2dzDAFfazDiUHFCZZRdWVRqP7l3L3aFSURPL9q"
access_token_secret = "SuW1MfXZtBB1JJ7iXcSuUA4jxlneMAQdPQzWKVo22NpS8"
consumer_key = "N2K1qgJDKqsYkvqNXEa1eNjOV"
consumer_secret = "Cx0bzYKvX8an4v2hPl4Sc1xnXVnLI8Hv7heg9IvscS2c82242a"

class StdOutListener(StreamListener):

    def on_data(self, data):
        print data
        return True

    def on_error(self, status):
        print status

if __name__ == '__main__':

    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    stream.filter(track=['python'])
