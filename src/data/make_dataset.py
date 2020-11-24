# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from twython import Twython



ACCESS_TOKEN = "761441357315440640-suCCQJo6kuufi3PmcYUl2y9kNyYb8C0"
ACCESS_TOKEN_SECRET = "nN4nX0LhlUZHN31LLYU1neOxg7elvb4LIo9KkX7gMDMaN"
API_KEY = "oMlZlYVi6MerYj7SZzcYWvgVr"
API_SECRET_KEY = "OW8cYRS69LUQ1gD5rKULGi4QtuBoj0OX5hRyJI5HVBbzTLZzam"



#@click.command()
#@click.argument('input_filepath', type=click.Path(exists=True))
#@click.argument('output_filepath', type=click.Path())

def collect_tweets(query='', geocode=None, result_type='recent',
                   num_of_page=20, count=100, since=None, until=None):
    '''Collects a number of tweets using Twitter standard search API and 
    returns a list of dictionaries each representing a tweet.

    query: search query
    geocode: Returns tweets by users located within a given radius 
             of the given lat/long. The parameter value is specified 
             by " latitude,longitude,radius "
    result_type: Specifies what type of search results you would prefer to receive. 
                  mixed : Include both popular and real time results in the response.
                  recent : return only the most recent results in the response
                  popular : return only the most popular results in the response.
    num_of_page: number of pages to collect.
    count: The number of tweets to return per page, up to a maximum of 100. 
           Defaults to 15.
    since: Returns tweets created after the given date. 
           Date should be formatted as YYYY-MM-DD. 
           The search index has a 7-day limit.
    until: Returns tweets created before the given date. 
           Date should be formatted as YYYY-MM-DD. 
           The search index has a 7-day limit.
    since_id: Returns results with an ID greater than 
              (that is, more recent than) the specified ID. 
              There are limits to the number of Tweets which 
              can be accessed through the API. If the limit of 
              Tweets has occured since the since_id, the since_id 
              will be forced to the oldest ID available.
    max_id: Returns results with an ID less than 
            (that is, older than) or equal to the specified ID.
    include_entities: The entities node will not be included when set to false.
    '''

    # Authentication
    twitter_obj = Twython(API_KEY, API_SECRET_KEY,
                          ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

    # Use Twitter standard API search
    tweet_result = twitter_obj.search(q=query, geocode=geocode,
                                      result_type=result_type, count=count,
                                      since=since, until=until,
                                      include_entities='true',
                                      tweet_mode='extended', lang='en')

    # In order to prevent redundant tweets explained here
    # https://developer.twitter.com/en/docs/tweets/timelines/guides/working-with-timelines
    # instead of reading a timeline relative to the top of the list
    # (which changes frequently), an application should read the timeline
    # relative to the IDs of tweets it has already processed.
    tweets_list = tweet_result['statuses']
    i = 0  # num of iteration through each page
    rate_limit = 1  # There is a limit of 100 API calls in the hour
    while tweet_result['statuses'] and i < num_of_page:
        if rate_limit < 1:
            # Rate limit time out needs to be added here in order to
            # collect data exceeding available rate-limit
            print(str(rate_limit)+' Rate limit!')
            break
        max_id = tweet_result['statuses'][len(
            tweet_result['statuses']) - 1]['id']-1

        tweet_result_per_page = twitter_obj.search(q=query, geocode=geocode,
                                                   result_type=result_type,
                                                   count=count, since=since,
                                                   until=until,
                                                   include_entities='true',
                                                   tweet_mode='extended',
                                                   lang='en',
                                                   max_id=str(max_id))

        tweets_list += tweet_result_per_page['statuses']
        i += 1
        rate_limit = int(twitter_obj.get_lastfunction_header(
            'x-rate-limit-remaining'))

    return tweets_list


def find_hashtags(tweet):
    hashtags = ''
    for i, term in enumerate(tweet):
        hashtags += term['text']+','
    return hashtags

def make_dataframe(tweets_list, search_term):
    '''Gets the list of tweets and return it as a pandas DataFrame.
    '''

    df = pd.DataFrame()
    df['tweet_id'] = list(map(lambda tweet: tweet['id'],
                              tweets_list))
    df['user'] = list(map(lambda tweet: tweet['user']
                          ['screen_name'], tweets_list))
    df['time'] = list(map(lambda tweet: tweet['created_at'], tweets_list))
    df['tweet_text'] = list(map(lambda tweet: tweet['full_text'], tweets_list))
    df['location'] = list(
        map(lambda tweet: tweet['user']['location'], tweets_list))
    df['hashtags'] = list(
        map(lambda tweet: find_hashtags(tweet['entities']['hashtags']), tweets_list))
    df['search_term'] = list(map(lambda tweet: search_term if search_term.lower(
    ) in tweet['full_text'].lower() else None, tweets_list))

    return df


#def main(input_filepath, output_filepath):
def main():
    """ Runs data processing scripts to collect data from twitter and turn raw data
        into cleaned data ready to be analyzed (saved in ../../data/processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    query = input('Enter a search word (Example: Walmart): ')
    #North America 49.525238,-93.874023,4000km
    tweets_list = collect_tweets(query=query, geocode="49.525238,-93.874023,4000km")
    df = make_dataframe(tweets_list, query)
    df.to_csv('../../data/processed/tweets.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    #load_dotenv(find_dotenv())

    main()
