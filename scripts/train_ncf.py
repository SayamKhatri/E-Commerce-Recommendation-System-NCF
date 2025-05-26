from src.common.data_loader import load_data, encode_data
from src.common.config import DATA_PATH, NCF_PARAMS
from src.NCF.preprocess import leave_one_out_split, negative_sampling
from src.NCF.model import NCFModel
from src.NCF.train import train_ncf
from src.NCF.evaluate import evaluate_metrics

def main():
    df = load_data(DATA_PATH)
    df, _, _ = encode_data(df)
    train_data, test_data = leave_one_out_split(df)
    train_samples = negative_sampling(train_data, train_data['visitorid_encoded'].nunique(), 
                                      train_data['itemid_encoded'].nunique(), 
                                      NCF_PARAMS['neg_ratio'])
    
    val_data = train_data[train_data['interaction'] > 0][['visitorid_encoded', 'itemid_encoded', 'interaction']].sample(frac=0.1)

    model = NCFModel(train_data['visitorid_encoded'].nunique(), train_data['itemid_encoded'].nunique())
    history = train_ncf(model, train_samples, val_data)
    
    print(f"Final training MSE: {history.history['mean_squared_error'][-1]}")
    metrics = evaluate_metrics(model, test_data)
    model.save('models/ncf/ncf_model')

if __name__ == "__main__":
    main()