import logging
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
import re
import seaborn as sns
from twython import Twython
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation

STOP_WORDS = ["'d", "'ll", "'re", "'s", "'ve", 'could', 'might', 'must',
              "n't", 'need', 'sha', 'wo', 'would', "ca", "na", "rt", "like",
              'u', 'get', 'got']


def clean_tweet_text(tweet, user_flag=True, urls_flag=True,
                     punc_flag=True, number_flag=True,
                     special_char_flag=True,
                     stop_word_flag=False):
    '''Clean a tweet by performing the following.

    - Remove username
    - Remove urls
    - Remove all punctuation and special character
    - Remove all stopwords if flag is True
    - Returns a cleaned text
    '''

    # remove the user
    if user_flag:
        tweet = re.sub(r'@[w\w]+', ' ', tweet)

    # remove the urls
    if urls_flag:
        tweet = re.sub(
            r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', ' ', tweet)

    # replace the negations
    tweet = re.sub(r"n't", ' not', tweet)
    tweet = re.sub(r"N'T", ' NOT', tweet)

    # remove punctuations
    if punc_flag:
        tweet = re.sub(
            '[!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@’…”'']+', ' ', tweet)

    # remove numbers
    if number_flag:
        tweet = re.sub('([0-9]+)', '', tweet)

    # remove special characters
    if special_char_flag:
        tweet = re.sub(r'[^\w]', ' ', tweet)

    # remove double space
    tweet = re.sub('\s+', ' ', tweet)

    if stop_word_flag:
        tweet = ' '.join([word for word in tweet.split() if word.lower(
        ) not in stopwords.words('english')+STOP_WORDS])

    return tweet


############# NLP #############3

def tokenize_tweet(tweet):
    '''Convert the normal text strings in to a list of tokens(words)
    '''
    # Tokenization
    return [word for word in tweet.split()]


def create_bag_of_words(tweets, max_df=1.0, min_df=1,
                        max_features=None):
    '''Vectorize the tweets using bag of words.
    
    Return the vectorized/bag of words of tweets
    as well as the features' name.
    
    max_df: float in range [0.0, 1.0] or int, default=1.0
            When building the vocabulary ignore terms that 
            have a document frequency strictly higher than 
            the given threshold (corpus-specific stop words). 
            If float, the parameter represents a proportion 
            of documents, integer absolute counts. 
            This parameter is ignored if vocabulary is not None.

    min_df: float in range [0.0, 1.0] or int, default=1
            When building the vocabulary ignore terms that 
            have a document frequency strictly lower than 
            the given threshold. This value is also called 
            cut-off in the literature. If float, the parameter 
            represents a proportion of documents, integer absolute 
            counts. This parameter is ignored if vocabulary is not None.
    '''

    # Vectorization using Countvectorize
    cv = CountVectorizer(analyzer=tokenize_tweet, max_df=max_df,
                         min_df=min_df, max_features=max_features)
    tweets_bow = cv.fit_transform(tweets)
    feature_names = cv.get_feature_names()
    return tweets_bow, feature_names


def create_tfidf(tweets_bow):
    '''Create the TF-IDF of tweets
    '''
    tfidf_transformer = TfidfTransformer().fit(tweets_bow)
    tweets_tfidf = tfidf_transformer.transform(tweets_bow)
    return tweets_tfidf


def sentiment_analysis(tweet):
    '''Takes a tweet and return a dictionary of scores in 4 categories.
    - Negative score
    - Neutral score
    - Positive score
    - Compound score
    
    Special characters and stopwords need to stay in the tweet.
    '''

    #create an instance of SentimentIntensityAnalyzer
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(tweet)


#sklearn.decomposition.LatentDirichletAllocation
def topic_modeling(tweets, num_components=7,
                   num_words_per_topic=10,
                   random_state=42,
                   max_df=1.0, min_df=1,
                   max_features=None):
    '''Get all the tweets and return the topics
    and the highest probability words per topic
    '''

    #create the bags of words
    tweets_bow, feature_names = create_bag_of_words(tweets, max_df=max_df,
                                                    min_df=min_df,
                                                    max_features=max_features)

    #create an instace of LatentDirichletAllocation
    lda = LatentDirichletAllocation(n_components=num_components,
                                    random_state=random_state)
    lda.fit(tweets_bow)

    #grab the highest probability words per topic
    words_per_topic = {}
    for index, topic in enumerate(lda.components_):
        words_per_topic[index] = [feature_names[i]
                                  for i in topic.argsort()[-15:]]

    topic_results = lda.transform(tweets_bow)
    topics = topic_results.argmax(axis=1)
    return topics, words_per_topic


def find_top_hashtags(hashtags):
    '''Get the hashtag column and find the popular hashtags.
    
    '''
    pass


def main():
    """ Runs NLP.
    """
    logger = logging.getLogger(__name__)
    logger.info('running setiment analysis')

    df = pd.read_csv('../../data/processed/tweets.csv')

    df['clean_text'] = df['tweet_text'].apply(
      lambda text: clean_tweet_text(text, punc_flag=False, 
      number_flag=False, special_char_flag=False))

    df['no_stop_words_text'] = df['tweet_text'].apply(
      lambda text: clean_tweet_text(text, stop_word_flag=True))

    df['sentiment_scores'] = df['clean_text'].apply(
      lambda tweet: sentiment_analysis(tweet))

    df['compound_scores'] = df['sentiment_scores'].apply(
      lambda scores_dict: scores_dict['compound'])


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
