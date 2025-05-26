import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    df = pd.read_parquet(file_path)
    df = df[df['event'].isin(['view', 'addtocart', 'transaction'])]
    
    return df 

def encode_data(df):
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    df['itemid_encoded'] = item_encoder.fit_transform(df['itemid'])
    df['visitorid_encoded'] = user_encoder.fit_transform(df['visitorid'])

    return df, user_encoder, item_encoder