from common.utils import compute_recall_at_k

def evaluate_metrics(model, test_data, k_values=[5, 10]):
    test_data = test_data.copy()
    predictions = model.predict([test_data['visitorid_encoded'].values, 
                               test_data['itemid_encoded'].values]).flatten()
    test_data['predicted_score'] = predictions
    
    print("\nTest Set Diagnostics:")
    print(f"Test rows: {len(test_data)}")
    print(f"Unique test users: {test_data['visitorid_encoded'].nunique()}")
    print(f"Average relevant items per user: {len(test_data[test_data['interaction'] >= 1]) / test_data['visitorid_encoded'].nunique()}")
    
    recall_scores = {k: [] for k in k_values}
    map_scores = {k: [] for k in k_values}
    
    for user_id in test_data['visitorid_encoded'].unique():
        user_data = test_data[test_data['visitorid_encoded'] == user_id]
        user_data = user_data.sort_values('predicted_score', ascending=False)
        top_items = user_data['itemid_encoded'].values
        top_interactions = user_data['interaction'].values
        relevant_items = user_data[user_data['interaction'] >= 1]['itemid_encoded'].values
        if not relevant_items:
            continue
        for k in k_values:
            recall_scores[k].append(compute_recall_at_k(top_items, relevant_items, k))
            ap = 0
            hits = 0
            for i in range(min(k, len(top_interactions))):
                if top_interactions[i] >= 1:
                    hits += 1
                    ap += hits / (i + 1)
            if hits > 0:
                map_scores[k].append(ap / hits)
    
    for k in k_values:
        recall = sum(recall_scores[k]) / len(recall_scores[k]) if recall_scores[k] else 0.0
        map_k = sum(map_scores[k]) / len(map_scores[k]) if map_scores[k] else 0.0
        print(f"Recall@{k}: {recall:.4f}")
        print(f"MAP@{k}: {map_k:.4f}")
    
    return {'recall': {k: recall for k in k_values}, 'map': {k: map_k for k in k_values}}