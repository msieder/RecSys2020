import pandas as pd
import numpy as np

def load_inital_dataframe(filepath:str) -> pd.DataFrame:
    column_names = ["text_tokens", "hashtags", "tweet_id", "present_media", "present_links", "present_domains",\
                "tweet_type", "language", "tweet_timestamp", "engaged_with_user_id", "engaged_with_user_follower_count",\
               "engaged_with_user_following_count", "engaged_with_user_is_verified", "engaged_with_user_account_creation",\
               "engaging_user_id", "engaging_user_follower_count", "engaging_user_following_count", "engaging_user_is_verified",\
               "engaging_user_account_creation", "engaged_follows_engaging", "reply_timestamp", "retweet_timestamp", "retweet_with_comment_timestamp", "like_timestamp"]

    df = pd.read_csv(filepath, header=None, names=column_names, delimiter='\x01')

    return df



def load_subsample(filepath:str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    df['text_tokens'] = df['text_tokens'].str.replace('\t', ' ')
    df['hashtags'] = df['hashtags'].fillna('')
    df['hashtags'] = df['hashtags'].str.replace('\t', ' ')


    def to_hex_list(x):
        output = str(x).split('\t')
    #     output = [int(val, 16) for val in str(x).split('\t')] 
        return output

    cols_to_process = ['present_media', 'present_links', 'present_domains']

    for col in cols_to_process:  
        df[col] = df[col].apply(lambda x: to_hex_list(x) if isinstance(x, str)  else x)
        
    cols_to_process = ['tweet_timestamp', 'engaging_user_account_creation', 'engaged_with_user_account_creation',  'reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']

    for col in cols_to_process:  
        df[col] = df[col].apply(lambda x: pd.Timestamp(x, unit='s'))
    
    return df
