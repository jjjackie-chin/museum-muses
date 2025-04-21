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
        description = dataset[dataset['MuseumName'] == name]['Description'].values[0]
        review_str = " ".join(review)
        content_str = " ".join(content)
        desc_str = description if isinstance(description, str) else ""
        review_texts.append(review_str + " " + content_str + " " + desc_str) 

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

    print("museum_names  in sims.py")
    
    # Check if we have enough data
    if len(review_texts) == 0:
        print("ERROR: No review texts available after filtering")
        return []
    
    print(f"Processing {len(museum_names)} museums")
        
    # Add query to the texts for vectorization
    svd_texts = [query_text] + review_texts

    # svd_texts = review_texts
    
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.9)
    td_matrix = vectorizer.fit_transform(svd_texts)
    # k = min(100, min(td_matrix.shape) - 1) 
    k = 30
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

            # print(u.shape)
            # print(s.shape)
            # print(vt.shape)   

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
        if name not in filtered_museums:
            continue
        try:
            address = dataset[dataset['MuseumName'] == name]['Address'].values[0]
            matching.append((name, address, float(sim)))
        except Exception as e:
            print(f"Error getting address for museum {name}: {e}")
            matching.append((name, "Address not found", float(sim)))



    matching.sort(key=lambda x: x[2], reverse=True)
    return matching

