def check_overlap(train_data, test_data):
    train_pairs = set(zip(train_data['visitorid_encoded'], train_data['itemid_encoded']))
    test_pairs = set(zip(test_data['visitorid_encoded'], test_data['itemid_encoded']))
    return len(train_pairs.intersection(test_pairs))

def compute_recall_at_k(predictions, relevant_items, k):
    hits = sum(item in relevant_items for item in predictions[:k])
    return hits / len(relevant_items) if relevant_items else 0.0