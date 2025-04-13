import math
from helpers.data_cleaning import getDataset
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter
from scipy.sparse.linalg import svds
import numpy as np

def inv_idx(msgs):
    inverted_index = {}
    for doc_id, msg in enumerate(msgs):
        token_counts = {}
        for token in msg["toks"]:
            token_counts[token] = token_counts.get(token, 0) + 1
        for token, count in token_counts.items():
            if token not in inverted_index:
                inverted_index[token] = []
            inverted_index[token].append((doc_id, count))
    return inverted_index


def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
    idf = {}
    max_df = max_df_ratio * n_docs  
    for term, postings in inv_idx.items():
        doc_freq = len(postings)  
        if min_df <= doc_freq <= max_df:
            idf[term] = math.log2(n_docs / (doc_freq + 1))  
    return idf


def composeData(dataset, filtered_museums=None):
    """
    Helper function that returns a list of review texts composed of museum reviews and contents
    """
    if not filtered_museums:
        museum_names = dataset['MuseumName'].tolist()
    else:
        museum_names = filtered_museums

    review_texts = []
    for name in museum_names:
        review = dataset[dataset['MuseumName'] == name]['Reviews'].values[0]
        content = dataset[dataset['MuseumName'] == name]['Content'].values[0]
        review_str = " ".join(review)
        content_str = " ".join(content)
        review_texts.append(review_str + " " + content_str) 

    return museum_names, review_texts




from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def SimGetMuseums(input_query, filtered_museums=None):
    """
    Takes in the query from the user and returns a list of all 
    museum names that are similar to the query using cosine similarity

    Returns:
    matching: list[str] list of museum names sorted by similarity scores
    """
    dataset = getDataset()

    vectorizer = TfidfVectorizer()
    tokenizer = TreebankWordTokenizer()
    query_tokens = tokenizer.tokenize(input_query.lower())
    query_text = " ".join(query_tokens)
    museum_names, review_texts = composeData(dataset)
    all_texts = [query_text] + review_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    query_vector = tfidf_matrix[0:1]  
    review_vectors = tfidf_matrix[1:]  
    
    similarities = cosine_similarity(query_vector, review_vectors)[0]
    
    # Get (name, address, similarity) tuples
    matching = []
    for name, sim in zip(museum_names, similarities):
        address = dataset[dataset['MuseumName'] == name]['Address'].values[0]
        matching.append((name, address, sim))
    
    # Sort by similarity
    matching.sort(reverse=True, key=lambda x: x[2])
    return matching

def SVDTopMuseums(input_query, filtered_museums=None):
    print("Starting SVDTopMuseums with query:", input_query)
    # print("Filtered museums:", filtered_museums)
    
    dataset = getDataset()
    # print("Dataset loaded, shape:", dataset.shape if hasattr(dataset, 'shape') else "unknown shape")
    
    tokenizer = TreebankWordTokenizer()
    query_tokens = tokenizer.tokenize(input_query.lower())
    query_text = " ".join(query_tokens)
    # print("Processed query:", query_text)
    
    museum_names, review_texts = composeData(dataset)
    # print(f"Composed data: {len(museum_names)} museums, {len(review_texts)} review texts")
    
    if filtered_museums is not None:
        indices = [i for i, name in enumerate(museum_names) if name in filtered_museums]
        museum_names = [museum_names[i] for i in indices]
        review_texts = [review_texts[i] for i in indices]
    # print(filtered_museums)
    
    # Check if we have enough data
    if len(review_texts) == 0:
        print("ERROR: No review texts available after filtering")
        return []
    
    print(f"Processing {len(museum_names)} museums")
    
    # print(f"Sample review text (first 100 chars): {review_texts[0][:100]}...")
    
    # Add query to the texts for vectorization
    all_texts = [query_text] + review_texts
    print(f"Created all_texts with {len(all_texts)} items")
    
    vectorizer = TfidfVectorizer(min_df=1)
    td_matrix = vectorizer.fit_transform(all_texts)
    # print("TF-IDF matrix shape:", td_matrix.shape)
    # print("Vocabulary size:", len(vectorizer.vocabulary_))
    
    k = min(3, min(td_matrix.shape) - 1)    
    if k <= 0:
        print("ERROR: k <= 0, cannot perform SVD")
        query_vector = td_matrix[0:1]
        docs_vectors = td_matrix[1:]
        similarities = (query_vector @ docs_vectors.T).toarray().flatten()
    else:
        # Apply SVD
        try:
            print("SVD...")
            from scipy.sparse.linalg import svds
            import numpy as np
            
            u, s, vt = svds(td_matrix, k=k)            
            s_diag = np.diag(s)
            docs_transformed = np.dot(u, s_diag)            
            query_vec = docs_transformed[0]
            docs_vecs = docs_transformed[1:]
            
            from sklearn.preprocessing import normalize
            docs_vecs_norm = normalize(docs_vecs)
            query_vec_norm = query_vec / np.linalg.norm(query_vec)            
            similarities = np.dot(docs_vecs_norm, query_vec_norm)

        except Exception as e:
            print(f"SVD calculation failed with error: {e}")
            query_vector = td_matrix[0:1]
            docs_vectors = td_matrix[1:]
            similarities = (query_vector @ docs_vectors.T).toarray().flatten()
    
    matching = []
    for i, (name, sim) in enumerate(zip(museum_names, similarities)):
        try:
            address = dataset[dataset['MuseumName'] == name]['Address'].values[0]
            matching.append((name, address, float(sim)))
        except Exception as e:
            print(f"Error getting address for museum {name}: {e}")
            matching.append((name, "Address not found", float(sim)))
    
    matching.sort(key=lambda x: x[2], reverse=True)
    return matching


# For testing purposes
# from data_cleaning import getDataset, filterCategory, filterLocation

# query = "Children"
# categories = ["Art Museums", "History Museums"]
# locations = ['NY']
# filtered_museums = set(filterCategory(categories))
# filtered_museums &= set(filterLocation(locations))
# SVDTopMuseums(query, filtered_museums=filtered_museums)