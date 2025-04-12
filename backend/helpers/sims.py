
import math
from helpers.data_cleaning import getDataset
from nltk.tokenize import TreebankWordTokenizer
from collections import Counter

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
    if not filtered_museums:
        museum_names = dataset['MuseumName'].tolist()
    else:
        museum_names = filtered_museums
    tokenizer = TreebankWordTokenizer()

    query_tokens = tokenizer.tokenize(input_query.lower())
    query_text = " ".join(query_tokens)
    
    review_texts = []
    for name in museum_names:
        review = dataset[dataset['MuseumName'] == name]['Reviews'].values[0]
        review_str = " ".join(review)
        review_texts.append(review_str)
    
    vectorizer = TfidfVectorizer()

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
