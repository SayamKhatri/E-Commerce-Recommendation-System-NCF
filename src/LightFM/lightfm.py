import pandas as pd 
import sys 
import os

sys.path.append(os.path.abspath(os.path.join('..')))

from common.data_loader import load_data

df = load_data('/Users/samkhatri/Desktop/Data Science Projects/E-Commerce-Recommendation-System-NCF/data/df_ncf_events.parquet')

print(df.head())