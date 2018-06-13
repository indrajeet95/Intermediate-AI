import json
import pandas as pd
import matplotlib.pyplot as plt
import re

def main():
    #Reading Tweets
    print 'Reading Tweets\n'
    tweets_data_path = 'tweets_2.txt'

    tweets_data = []
    tweets_file = open(tweets_data_path, "r")
    for line in tweets_file:
        try:
            tweet = json.loads(line)
            tweets_data.append(tweet)
        except:
            continue

    #Structuring Tweets
    print 'Structuring Tweets\n'
    tweets = pd.DataFrame()
    tweets['lang'] = map(lambda tweet: tweet['lang'], tweets_data)
    #Analyzing Tweets by Language
    print 'Analyzing tweets by language\n'
    print len(tweets)
    tweets_by_lang = tweets['lang'].value_counts()
    print (tweets_by_lang/len(tweets))*100

    print 'Langid.py supports 97 languages' # https://github.com/saffsd/langid.py/blob/master/README.rst
    fig, ax = plt.subplots()
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=10)
    ax.set_xlabel('Languages', fontsize=15)
    ax.set_ylabel('Number of tweets' , fontsize=15)
    ax.set_title('Top 5 languages', fontsize=15, fontweight='bold')
    tweets_by_lang[:5].plot(ax=ax, kind='bar', color='red')
    plt.savefig('tweet_by_lang', format='png')

if __name__=='__main__':
    main()
