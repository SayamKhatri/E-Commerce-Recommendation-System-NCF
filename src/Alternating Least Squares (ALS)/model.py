from scipy.sparse import coo_matrix
from preprocess import preprocess
import implicit 


def train_model():

    agg_events, user_to_idx, item_to_idx, test_events = preprocess()

    # ALS needs a compact range of integers, that is why we map each id to a new continous index 

    unique_users = agg_events['visitorid'].unique().tolist()
    unique_items = agg_events['itemid'].unique().tolist()

    row = agg_events['item_idx'].values
    col = agg_events['user_idx'].values
    data = agg_events['confidence'].values

    matrix = coo_matrix(
        (data, (row, col)) , 
        shape = (len(unique_items) , len(unique_users))
    )

    train_csr = matrix.T.tocsr()

    # Recomendations using ALS

    model = implicit.als.AlternatingLeastSquares(
        factors = 150,
        regularization = 1.0 ,
        iterations = 50
        
    )

    model.fit(train_csr)


    idx_to_item = {}
    for key, value in item_to_idx.items():
        idx_to_item[value] = key

    return model, user_to_idx, idx_to_item, test_events, train_csr