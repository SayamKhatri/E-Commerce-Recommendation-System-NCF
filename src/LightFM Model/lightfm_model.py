import pandas as pd 
import sys 
import os
from lightfm import LightFM
from lightfm.data import Dataset

sys.path.append(os.path.abspath(os.path.join('..')))

from Alternating_Least_Squares.preprocess import preprocess
from common.data_loader import load_data
from common.config import ITEM_PATH

_, user_to_idx, item_to_idx, test_events, train_events = preprocess()

baseline_item_feats = load_data(ITEM_PATH)
print(baseline_item_feats.head())




