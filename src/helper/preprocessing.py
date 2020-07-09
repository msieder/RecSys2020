from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, KBinsDiscretizer
from sklearn.feature_extraction.text import TfidfVectorizer

import scipy.sparse as sp
import pickle

import numpy as np


def preprocess_dataset(df, pipeline_path, load=False) -> (np.ndarray, np.ndarray):

    if load:
        with open(pipeline_path, "rb") as file:
            components = pickle.load(file)

    # 1)
    reply = df["reply_timestamp"].notnull().astype(int).to_numpy()
    retweet = df["retweet_timestamp"].notnull().astype(int).to_numpy()
    retweet_with_comment = df["retweet_with_comment_timestamp"].notnull().astype(int).to_numpy()
    like = df["like_timestamp"].notnull().astype(int).to_numpy()

    response = np.hstack((reply, retweet, retweet_with_comment,like))

    # 2)
    if load:
        language = components['language_encoder'].transform(df["language"].to_numpy().reshape(-1,1))
        tweet_type = components["tweet_type_encoder"].transform(df["tweet_type"].to_numpy().reshape(-1,1))
        present_media = components["present_media_encoder"].transform(df["present_media"])
    else:
        language_encoder = OneHotEncoder()
        language = language_encoder.fit_transform(df["language"].to_numpy().reshape(-1,1))
        tweet_type_encoder = OneHotEncoder()
        tweet_type = tweet_type_encoder.fit_transform(df["tweet_type"].to_numpy().reshape(-1,1))
        present_media_encoder = MultiLabelBinarizer(sparse_output=False)
        present_media = present_media_encoder.fit_transform(df["present_media"])

    tweet_features = sp.hstack([language, tweet_type, present_media])

    #3)
    if load:
        text_tokens = components["text_tfidf"].transform(df['text_tokens'])
    else:
        text_tfidf = TfidfVectorizer()
        text_tokens = text_tfidf.fit_transform(df['text_tokens'])

    #4) 
    if load:
        hashtags = components["hashtags_tfidf"].transform(df['hashtags'])
    else:
        hashtags_tfidf = TfidfVectorizer()
        hashtags = hashtags_tfidf.fit_transform(df['hashtags'])

    tweet_features = sp.hstack((text_tokens,hashtags))  # NOT np.vstack
    # 5)
    if load:
        df['tweet_id'] = df["tweet_id"].map(hash)
        df['engaged_with_user_id'] = df["engaged_with_user_id"].map(hash)
        df['engaging_user_id'] = df["engaging_user_id"].map(hash)

        tweet_id = components["tweet_discretizer"].transform(df['tweet_id'].to_numpy().reshape(-1, 1))
        engaged_with_user_id = components["engaged_with_user_discretizer"].transform(df['engaged_with_user_id'].to_numpy().reshape(-1, 1))
        engaging_user_id = components["engaging_user_discretizer"].transform(df['engaging_user_id'].to_numpy().reshape(-1, 1))
    else:
        df['tweet_id'] = df["tweet_id"].map(hash)
        df['engaged_with_user_id'] = df["engaged_with_user_id"].map(hash)
        df['engaging_user_id'] = df["engaging_user_id"].map(hash)

        tweet_discretizer = KBinsDiscretizer(n_bins=50)
        tweet_id = tweet_discretizer.fit_transform(df['tweet_id'].to_numpy().reshape(-1, 1))
        engaged_with_user_discretizer = KBinsDiscretizer(n_bins=50)
        engaged_with_user_id = tweet_discretizer.fit_transform(df['engaged_with_user_id'].to_numpy().reshape(-1, 1))
        engaging_user_discretizer = KBinsDiscretizer(n_bins=50)
        engaging_user_id = tweet_discretizer.fit_transform(df['engaging_user_id'].to_numpy().reshape(-1, 1))

    id_features = sp.hstack([tweet_id, engaged_with_user_id, engaging_user_id])

    # 6)
    engaged_with_user_is_verified = df["engaged_with_user_is_verified"].astype(int).to_numpy()
    engaging_user_is_verified = df["engaging_user_is_verified"].astype(int).to_numpy()
    engaged_follows_engaging = df["engaged_follows_engaging"].astype(int).to_numpy()
    boolean_features = np.column_stack([engaged_with_user_is_verified,engaging_user_is_verified, engaged_follows_engaging ])

    # 7)
    present_links = df["present_links"].notnull().astype(int).to_numpy()
    present_domains = df["present_domains"].notnull().astype(int).to_numpy()

    present_features = np.column_stack([present_links,present_domains ])

    X_train = sp.hstack([tweet_features, id_features, boolean_features, present_features])

    if not load:
        components = {
            "language_encoder": language_encoder,
            "tweet_type_encoder": tweet_type_encoder,
            "present_media_encoder": present_media_encoder,
            "text_tfidf": text_tfidf,
            "hashtags_tfidf": hashtags_tfidf,
            "tweet_discretizer": tweet_id,
            "engaged_with_user_discretizer": engaged_with_user_discretizer,
            "engaging_user_discretizer": engaging_user_discretizer
        }
        with open(pipeline_path, "wb") as file:
            pickle.dump(components, file)

    return X_train, response