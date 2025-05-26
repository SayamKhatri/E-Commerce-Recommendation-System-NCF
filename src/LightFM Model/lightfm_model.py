import pandas as pd 
import sys 
import os
from lightfm import LightFM
from lightfm.data import Dataset
sys.path.append(os.path.abspath(os.path.join('..')))

from Alternating_Least_Squares.preprocess import preprocess
from common.data_loader import load_data
from common.config import ITEM_PATH

from tqdm import tqdm
import numpy as np


_, user_to_idx, item_to_idx, test_events, train_events = preprocess()

baseline_item_feats = load_data(ITEM_PATH)

train_events_item = train_events['itemid'].unique()
baseline_item_feats_filtered = baseline_item_feats[baseline_item_feats['itemid'].isin(train_events_item)]
baseline_item_feats_filtered.reset_index(inplace = True, drop = True)

train_events_filtered = train_events[train_events['itemid'].isin(baseline_item_feats_filtered['itemid'].unique())]
train_events_filtered.reset_index(inplace = True, drop = True)

agg_events_filtered = train_events_filtered.groupby(['visitorid' , 'itemid'])['confidence'].sum().reset_index()

cats = []
for cat in baseline_item_feats_filtered['categoryid'].unique():
    cats.append(f"cat:{cat}")

feature_vocab = ['available'] + cats

ds = Dataset(item_identity_features=False)

ds.fit(
    users = agg_events_filtered['visitorid'].unique(),
    items = agg_events_filtered['itemid'].unique(),
    user_features=None,
    item_features = feature_vocab
    
)

interactions = list(
    zip(
        agg_events_filtered['visitorid'],
        agg_events_filtered['itemid'],
        agg_events_filtered['confidence']
        
    )
)

interaction_matrix, weights_matrix = ds.build_interactions(interactions)


idx_to_item = {}
for key, value in item_to_idx.items():
    idx_to_item[value] = key



item_features_list = []
for _,row in baseline_item_feats_filtered.iterrows():
    item_id = row['itemid']
    cat = 'cat:' + str(row['categoryid'])
    item_features_list.append((item_id, ['available', cat]))

item_features = ds.build_item_features(item_features_list)
print(item_features.shape)

hybrid = LightFM(
    no_components=50,
    loss='warp',
    learning_rate=0.05,

)

hybrid.fit(
    interaction_matrix,
    item_features = item_features,
    sample_weight = weights_matrix,
    epochs = 30,
    num_threads = 4
)


user_map, _, item_map, _ = ds.mapping()

idx_to_item_lightfm = {v:k for k,v in item_map.items()}

K = 10
hits = 0
evaluated = 0

for real_user in tqdm(test_events['visitorid'].unique(), desc="Evaluating Users"):

    if real_user not in user_map:
        continue
    evaluated += 1

    uidx = user_map[real_user]


    n_items = interaction_matrix.shape[1]
    scores = hybrid.predict(
        uidx,                         
        np.arange(n_items),            
        item_features=item_features    
    )


    top_k_idx = np.argsort(-scores)[:K]


    recommendations = [idx_to_item[i] for i in top_k_idx]

 
    true_item = test_events.loc[
        test_events['visitorid']==real_user, 'itemid'
    ].iloc[0]

 
    if true_item in recommendations:
        hits += 1

recall_at_k = hits / evaluated
print(f"\nHybrid Recall@{K}: {recall_at_k:.4f}")


