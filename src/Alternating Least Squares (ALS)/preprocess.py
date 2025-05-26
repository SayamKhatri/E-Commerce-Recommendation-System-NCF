import pandas as pd
import sys 
import os 

sys.path.append(os.path.abspath(os.path.join('..')))

from common.data_loader import load_data
from common.config import DATA_PATH

def preprocess():

    df_events = load_data(DATA_PATH)
    # Filtering out users who only have one interaction in our raw data

    interactions_per_user = df_events['visitorid'].value_counts()
    reqd_users = interactions_per_user[interactions_per_user > 1].index.tolist()
    df_events = df_events[df_events['visitorid'].isin(reqd_users)]
    df_events.reset_index(inplace = True, drop = True)

    # Taking out the last interaction per user as the test_event

    sorted_df = df_events.sort_values(by = 'timestamp')
    last_per_user = sorted_df.groupby('visitorid').tail(1)
    test_events = last_per_user.copy()
    train_events = df_events.drop(index = last_per_user.index)
    print('Length of Train Events:', len(train_events) , '\n' , 'Length of Test Events:', len(test_events))
    train_events.reset_index(inplace = True, drop = True)

    # Each event should carry different weight
    weights = {
        'view' : 1.0 ,
        'addtocart' : 3.0,
        'transaction' : 5.0
    }

    train_events['confidence'] = train_events['event'].map(weights)

    agg_events = train_events.groupby(['visitorid' , 'itemid'])['confidence'].sum().reset_index()

    # ALS needs a compact range of integers, that is why we map each id to a new continous index 

    unique_users = agg_events['visitorid'].unique().tolist()
    unique_items = agg_events['itemid'].unique().tolist()

    user_to_idx = {}
    for new_index,user_id in enumerate(unique_users):
        user_to_idx[user_id] = new_index

    item_to_idx = {}
    for new_index, item_id in enumerate(unique_items):
        item_to_idx[item_id] = new_index
        

    # Adding our created index to the df
    agg_events['user_idx'] = agg_events['visitorid'].apply(lambda x : user_to_idx[x])
    agg_events['item_idx'] = agg_events['itemid'].apply(lambda x : item_to_idx[x])

    return agg_events, user_to_idx, item_to_idx, test_events
