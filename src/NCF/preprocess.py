import pandas as pd
import numpy as np
from common.utils import check_overlap

def leave_one_out_split(df, timestamp_col='timestamp'):
    valid_test_users = df['visitorid_encoded'].value_counts()[lambda x: x > 1].index
    df = df.sort_values(['visitorid_encoded', timestamp_col])
    test_idx = df[df['visitorid_encoded'].isin(valid_test_users)].groupby('visitorid_encoded').tail(1).index
    test_data = df.loc[test_idx]
    train_data = df.drop(test_idx)
    test_data = test_data[
        test_data['visitorid_encoded'].isin(train_data['visitorid_encoded']) &
        test_data['itemid_encoded'].isin(train_data['itemid_encoded'])
    ]
    print(f"Train-test overlap: {check_overlap(train_data, test_data)} pairs")
    
    return train_data, test_data

def negative_sampling(train_data, num_users, num_items, neg_ratio=1.0):
    positive_samples = train_data[['visitorid_encoded', 'itemid_encoded', 'interaction']].copy()

    num_positive = len(positive_samples)
    num_negative = int(num_positive * neg_ratio)
    negative_users = np.random.choice(train_data['visitorid_encoded'].unique(), size=num_negative, replace=True)
    negative_items = np.random.choice(train_data['itemid_encoded'].unique(), size=num_negative, replace=True)

    negative_samples = pd.DataFrame({
        'visitorid_encoded': negative_users,
        'itemid_encoded': negative_items,
        'interaction': np.zeros(num_negative)
    })

    train_positive_pairs = set(zip(train_data['visitorid_encoded'], train_data['itemid_encoded']))
    negative_samples = negative_samples[
        ~negative_samples.apply(
            lambda x: (x['visitorid_encoded'], x['itemid_encoded']) in train_positive_pairs, axis=1
        )
    ].head(num_negative)

    return pd.concat([positive_samples, negative_samples], ignore_index=True)