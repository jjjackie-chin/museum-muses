import pickle
from sentence_transformers import SentenceTransformer
from data_cleaning import getDataset

model = SentenceTransformer('all-MiniLM-L6-v2')
dataset = getDataset()

review_embeddings = {}

for name in dataset['MuseumName'].unique():
    reviews = dataset[dataset['MuseumName'] == name]['Reviews'].values[0]
    filtered = [r for r in reviews if 10 <= len(r) <= 300]
    if not filtered:
        continue
    try:
        embs = model.encode(filtered)
        review_embeddings[name] = (filtered, embs)
    except Exception as e:
        print(f"Error encoding reviews for {name}: {e}")

with open("cached_review_embeddings.pkl", "wb") as f:
    pickle.dump(review_embeddings, f)