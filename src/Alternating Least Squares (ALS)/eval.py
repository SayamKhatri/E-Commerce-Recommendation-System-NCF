from tqdm import tqdm
from model import train_model


def evaluate(model, user_to_idx, idx_to_item, test_events, train_csr, K):
    hits = 0
    evaulated = 0

    for user in tqdm(test_events['visitorid'].unique(), desc='Evaluating Users'):

        u_id = user_to_idx.get(user)
        user_vec = train_csr[u_id]
        evaulated += 1
        recos = model.recommend(
            u_id,
            user_vec,
            N = K,
            filter_already_liked_items=True
        )

        recomendaions = []

        for rec in recos[0]:
            item_idx = rec
            if item_idx in idx_to_item:
                real_item_idx = idx_to_item[item_idx]
                recomendaions.append(real_item_idx)


        true_item = test_events.loc[test_events['visitorid'] == user, 'itemid'].iloc[0]

        if true_item in recomendaions:
            hits +=1

    return hits/evaulated

if __name__ == "__main__": 
    K = 10
    model, user_to_idx, idx_to_item, test_events, train_csr = train_model()
    print(f'recall@{K}:', evaluate(model, user_to_idx, idx_to_item, test_events, train_csr, K))


