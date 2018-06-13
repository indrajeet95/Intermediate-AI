import sys
import matplotlib.pyplot as matplot
import numpy as np
import matplotlib.pyplot as matplot
import numpy as np

try:
    import json
except ImportError:
    import simplejson as json

try:
    from langid.langid import LanguageIdentifier, model
except:
    print "Cannot find LangID"
    sys.exit()

from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream

ACCESS_TOKEN = "76254882-igW2dzDAFfazDiUHFCZZRdWVRqP7l3L3aFSURPL9q"
ACCESS_SECRET = "SuW1MfXZtBB1JJ7iXcSuUA4jxlneMAQdPQzWKVo22NpS8"
CONSUMER_KEY = "N2K1qgJDKqsYkvqNXEa1eNjOV"
CONSUMER_SECRET = "Cx0bzYKvX8an4v2hPl4Sc1xnXVnLI8Hv7heg9IvscS2c82242a"

class Twitter_Language_Mix:
    lang_tagged_tweets = 0
    total_tweets = 0
    location_tweets = 0
    geo_tagged_tweets = 0

    lang_dict = {}
    langid_dict = {}
    disagree_dict = {}
    disagree_sample_dict = {}
    lang_percentage_dict = {}
    loc_lang_dict = {}
    und_lang_list = []
    agree_list = []
    disagree_list = []

    solutions = "solutions.txt"

    def process_general_tweets(self):
        general_tweets = "general_tweets.json"
        with open(general_tweets) as f:
            for line in f:
                tweet = json.loads(line.strip())
                if 'text' not in tweet:
                    continue
                self.total_tweets += 1
                self.check_language(tweet)
                if 'coordinates' in tweet and tweet['coordinates']:
                    self.geo_tagged_tweets += 1

        self.solution_file.write("Q1.\n")
        self.solution_file.write("Total number of tweets streamed: %s\n\n" % self.total_tweets)
        percent_lang_tagged = self.percentage(self.lang_tagged_tweets, self.total_tweets)
        self.solution_file.write("Q2.\n")
        self.solution_file.write("Number of Tweets lang-tagged by Twitter: %s (%s%%)\n" % (self.lang_tagged_tweets, percent_lang_tagged))
        self.solution_file.write("Number of different lang tags provided by Twitter: %s\n" % (len(self.lang_dict)))
        self.solution_file.write("Percentage of each language is plotted in LanguagePercentageDistribution.png\n\n")

        self.language_percentage(self.lang_dict, self.total_tweets)
        self.solution_file.write("Q3.\n")
        self.solution_file.write("Number of different languages tagged by LangID: %s\n" % len(self.langid_dict))
        self.solution_file.write("Percentage of Twitter and LangID tags that agree: %s%%\n\n" % self.percentage(len(self.agree_list), self.lang_tagged_tweets))

        self.solution_file.write("Top 5 languages they disagree on:\n")
        for tup in zip(sorted(self.disagree_dict, key=self.disagree_dict.get, reverse=True), sorted(self.disagree_dict.values(), reverse=True))[:5]:
            lang, cases = tup
            self.solution_file.write("%s: %s cases\n" % (lang, cases))
        self.solution_file.write('\n')

        self.solution_file.write("A sample of disagreed tweets in %s (English):\n\n" % (sorted(self.disagree_dict, key=self.disagree_dict.get, reverse=True)[0]))
        for tweet in self.disagree_sample_dict['en'][:5]:
            self.solution_file.write('"%s"\n' % tweet['text'].encode('utf-8'))
            self.solution_file.write("Twitter's Decision: %s\nLangId's Decision: %s \n\n" % (tweet['lang'], self.identifier.classify(tweet['text'])[0].encode('utf-8')))

        self.build_bar_plot(self.lang_percentage_dict, "Languages", "Percentage", "Language Percentage Distribution Across All Tweets.", "LanguagePercentDistribution.png")

    def process_usa_tweets(self):
        usa_tweets = "usa_tweets.json"
        self.solution_file.write("Q4.\n")
        self.solution_file.write("Number of tweets geotagged out of general tweets: %s (%s%%)\n\n" % (self.geo_tagged_tweets, self.percentage(self.geo_tagged_tweets, self.total_tweets)))
        total_lang_tagged = 0
        with open(usa_tweets) as f:
            for line in f:
                tweet = json.loads(line.strip())
                if 'text' not in tweet:
                    continue
                self.location_tweets += 1
                total_lang_tagged += 1
                lang = tweet['lang']
                if lang != "und":
                    self.loc_lang_dict[lang] = self.loc_lang_dict.get(lang, 0) + 1
                else:
                    self.und_lang_list.append(tweet['text'].encode('utf-8'))

        self.solution_file.write("Number of tweets from USA: %s\n" % total_lang_tagged)
        self.solution_file.write("Number of different languages in tweets from USA: %s\n" % len(self.loc_lang_dict))
        self.solution_file.write("Percentage of each language found in the USA is plotted in USLanguagePercentageDistribution.png\n\n")
        self.language_percentage(self.loc_lang_dict, total_lang_tagged)
        self.build_bar_plot(self.lang_percentage_dict, "Languages in USA", "Percentage", "Percentage Distribution of Languages in USA", "USLanguagePercentDistribution.png")
        self.loc_lang_dict = {}

    def build_bar_plot(self, data_dict, xlabel, ylabel, title, filename):
        matplot.rcdefaults()
        fig, ax = matplot.subplots(figsize=(14,8))
        y_list = sorted(data_dict, key = data_dict.get, reverse = True)
        x_list = sorted(data_dict.values(), reverse = True)
        bar_width = 15
        step = bar_width + 5
        y_range = np.arange(0, step * len(y_list), step)
        ax.bar(y_range, x_list, width = bar_width, fc = (0, 0, 1, 0.3), edgecolor='black', linewidth=2)
        ax.set_xticks(y_range + bar_width / 2)
        ax.set_xticklabels(y_list)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        matplot.savefig(filename)
        matplot.close()
        matplot.clf()

    def percentage(self, observed, total):
        return float(format((float(observed) / total) * 100, ".2f"))

    def language_percentage(self, lang_dict, total):
        self.lang_percentage_dict = {}
        for lang, count in lang_dict.iteritems():
            self.lang_percentage_dict[lang] = self.percentage(count, total)

    def check_language(self, tweet):
        tweet_text = tweet['text']
        langid_classified_lang = self.identifier.classify(tweet_text)[0]
        self.langid_dict[langid_classified_lang] = self.langid_dict.get(langid_classified_lang, 0) + 1

        if 'lang' in tweet:
            if tweet['lang'] == 'und':
                self.und_lang_list.append(tweet['text'].encode('utf-8'))
                return
            self.lang_tagged_tweets += 1
            language = tweet['lang']
            self.lang_dict[language] = self.lang_dict.get(language, 0) + 1
            if language != langid_classified_lang:
                self.disagree_list.append(tweet)
                self.disagree_dict[tweet['lang']] = self.disagree_dict.get(tweet['lang'], 0) + 1
                self.disagree_sample_dict.setdefault(tweet['lang'], []).append(tweet)
            else:
                self.agree_list.append(tweet)

    def run_main(self):
        self.solution_file = open(self.solutions, "w+")
        self.identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        self.process_general_tweets()
        self.process_usa_tweets()

solution = Twitter_Language_Mix()
solution.run_main()
