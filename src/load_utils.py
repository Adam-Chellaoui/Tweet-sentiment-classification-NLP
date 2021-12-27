import numpy as np
import pandas as pd


# Returns a dataframe
def load_tweets(full=False):
    """Load the tweets amd their labels from the .txt files into a dataframe

    Args:
        full (bool, optional): If True, then load the tweets fromm train_neg and train_pos, else use the _full versions. Defaults to False.

    Returns:
        [dataFrame]: a dataframe where tweets are on the "text" column and labels on the "label" column
    """

    if full == False:
        train_neg_r = open("twitter-datasets/train_neg.txt").readlines()
        train_pos_r = open("twitter-datasets/train_pos.txt").readlines()

    elif full == True:
        train_neg_r = open("twitter-datasets/train_neg_full.txt").readlines()
        train_pos_r = open("twitter-datasets/train_pos_full.txt").readlines()

    # create dataframe with positive tweets and "1" label
    pos_tr = pd.DataFrame()
    pos_tr['text'] = train_pos_r
    pos_tr['label'] = 1

    # create dataframe with negative tweets and "0" label
    neg_tr = pd.DataFrame()
    neg_tr['text'] = train_neg_r
    neg_tr['label'] = 0

    # concatenate dataframes
    tweets_df = pd.concat([pos_tr, neg_tr], ignore_index=True)
    tweets_df.index.name = 'id'

    print('loaded', len(tweets_df),
          'tweets in dataframe with columns:', tweets_df.columns)
    return tweets_df  # dataframe


def load_df(full=False, lemmatize=True):
    """Load the pre_cleaned dataframe.

    Args:
        full (bool, optional): If set to False, pick only 10% of the tweets. Defaults to False.
        lemmatize (bool, optional): If set to False, pick the non lemmatize version of tweets. Defaults to True.

    Returns:
        [dataFrame]: the dataframe of all tweets in the column "tweet", and labels in column "positive"
    """

    path = "../data/cleanedDataframe/cleanedDataframe"
    if(lemmatize):
        path = path + "_lemmatize"

    df = pd.read_pickle(path + ".pkl")

    if(not full):
        # select only 10% of the tweets
        df = df.sample(n=round(len(df)/10))
    return df
