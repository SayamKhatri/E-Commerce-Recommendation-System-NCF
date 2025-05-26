DATA_PATH = '/Users/samkhatri/Desktop/Data Science Projects/E-Commerce-Recommendation-System-NCF/data/df_ncf_events.parquet'
ITEM_PATH = '/Users/samkhatri/Desktop/Data Science Projects/E-Commerce-Recommendation-System-NCF/data/baseline_item_feats.parquet'
NCF_PARAMS = {
    'embedding_size': 64,
    'mlp_layers': [128, 64, 32],
    'learning_rate': 0.001,
    'epochs': 5,
    'neg_ratio': 1.0
}