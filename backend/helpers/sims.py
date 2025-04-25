import math
import pickle
from helpers.data_cleaning import getDataset
from nltk.tokenize import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
import numpy as np
import random
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# getting the cached reviews
try:
    with open("cached_review_embeddings.pkl", "rb") as f:
        all_review_embeddings = pickle.load(f)
except FileNotFoundError:
    all_review_embeddings = {}

# loading sbert
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')


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


def interpret_svd_dimensions(vt, feature_names):
    """Create interpretable labels for SVD dimensions based on top terms"""
    dimension_labels = []
    
    for i in range(vt.shape[0]):
        # Get indices of terms with highest absolute weights
        top_indices = np.abs(vt[i]).argsort()[-15:][::-1]
        top_terms = [feature_names[idx] for idx in top_indices]
        
        # Manually interpret based on top terms
        dimension_theme = interpret_dimension_theme(top_terms)
        dimension_labels.append(dimension_theme)
    
    return dimension_labels


def interpret_dimension_theme(top_terms):
    """Interpret theme based on top terms"""
    terms_str = ", ".join(top_terms[:7])
    
    # Museum-specific themes based on domain knowledge
    if any(term in top_terms for term in ["art", "painting", "sculpture", "gallery", "artist", "portrait", "artwork"]):
        return f"Art & Visual Culture ({terms_str})"
    elif any(term in top_terms for term in ["history", "historic", "war", "century", "historical", "civil", "ancient"]):
        return f"Historical Significance ({terms_str})"
    elif any(term in top_terms for term in ["science", "technology", "discovery", "scientific", "innovation", "engineering"]):
        return f"Science & Technology ({terms_str})"
    elif any(term in top_terms for term in ["kids", "children", "family", "interactive", "fun", "play", "educational"]):
        return f"Family Experience ({terms_str})"
    elif any(term in top_terms for term in ["nature", "animals", "wildlife", "garden", "natural", "outdoors", "ecosystem"]):
        return f"Natural World ({terms_str})"
    elif any(term in top_terms for term in ["exhibit", "collection", "gallery", "display", "artifacts", "pieces", "showcases"]):
        return f"Exhibition Quality ({terms_str})"
    elif any(term in top_terms for term in ["military", "war", "aviation", "naval", "army", "aircraft", "veterans"]):
        return f"Military History ({terms_str})"
    elif any(term in top_terms for term in ["architecture", "building", "design", "structure", "historic", "century", "beautiful"]):
        return f"Architecture & Design ({terms_str})"
    elif any(term in top_terms for term in ["cultural", "heritage", "tradition", "ethnic", "indigenous", "identity", "diversity"]):
        return f"Cultural Heritage ({terms_str})"
    elif any(term in top_terms for term in ["interactive", "hands-on", "experience", "engaging", "activities", "participatory"]):
        return f"Interactive Experience ({terms_str})"
    else:
        return f"Theme: {terms_str}"


def get_dominant_dimension(query_vec, doc_vec):
    """Find which dimension contributed most to the similarity"""
    # Element-wise product shows contribution of each dimension
    contributions = query_vec * doc_vec
    top_dim_index = np.argmax(np.abs(contributions))
    return top_dim_index, contributions[top_dim_index]


def get_representative_review(query, name, dataset):
    if name not in all_review_embeddings:
        return "No reviews available"

    reviews, review_vecs = all_review_embeddings[name]
    try:
        query_vec = sbert_model.encode([query])[0]
        sims = cosine_similarity([query_vec], review_vecs)[0]
        best_idx = sims.argmax()
        return reviews[best_idx]
    except:
        return reviews[0] if reviews else "No reviews available"


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

random.seed(42)
np.random.seed(42)
def SVDTopMuseums(input_query, filtered_museums=None):
    print("Starting SVDTopMuseums with query:", input_query)
    
    dataset = getDataset()
    
    tokenizer = TreebankWordTokenizer()
    query_tokens = tokenizer.tokenize(input_query.lower())
    query_text = " ".join(query_tokens)

    filtered_museums = sorted(filtered_museums) if filtered_museums else None
    museum_names, review_texts = composeData(dataset)

    
    # Check if we have enough data
    if len(review_texts) == 0:
        print("ERROR: No review texts available after filtering")
        return []
    
    print(f"Processing {len(museum_names)} museums")
        
    # Add query to the texts for vectorization
    svd_texts = [query_text] + review_texts
    
    vectorizer = TfidfVectorizer(min_df=1, max_df=0.9)
    td_matrix = vectorizer.fit_transform(svd_texts)
    num_docs = len(review_texts)
    k = min(30, max(2, num_docs // 2))
    
    if k <= 0:
        print("ERROR: k <= 0, cannot perform SVD")
        query_vector = td_matrix[0:1]
        docs_vectors = td_matrix[1:]
        similarities = (query_vector @ docs_vectors.T).toarray().flatten()
        
        matching = []
        for i, (name, sim) in enumerate(zip(museum_names, similarities)):
            if name not in filtered_museums:
                continue
            try:
                address = dataset[dataset['MuseumName'] == name]['Address'].values[0]
                review = get_representative_review(input_query, name, dataset)
                matching.append((name, address, float(sim), review, "No SVD used"))
            except Exception as e:
                print(f"Error getting data for museum {name}: {e}")
                matching.append((name, "Address not found", float(sim), "No review available", "Unknown dimension"))
    else:
        # Apply SVD
        try:
            print("SVD...")
            u, s, vt = svds(td_matrix, k=k)
            s_diag = np.diag(s)
            docs_transformed = np.dot(u, s_diag)            
            query_vec = docs_transformed[0]
            docs_vecs = docs_transformed[1:]
            
            from sklearn.preprocessing import normalize
            docs_vecs_norm = normalize(docs_vecs)
            query_vec_norm = query_vec / np.linalg.norm(query_vec)            
            similarities = np.dot(docs_vecs_norm, query_vec_norm)
            
            # if np.all(similarities <= 0):
            #     print("All similarities are negative. Falling back.")
            #     k_retry = max(2, k // 2)
            #     try:
            #         u, s, vt = svds(td_matrix, k=k_retry, solver='arpack')
            #         s_diag = np.diag(s)
            #         docs_transformed = np.dot(u, s_diag)            
            #         query_vec = docs_transformed[0]
            #         docs_vecs = docs_transformed[1:]
            #         docs_vecs_norm = normalize(docs_vecs)
            #         query_vec_norm = query_vec / np.linalg.norm(query_vec)
            #         similarities = np.dot(docs_vecs_norm, query_vec_norm)
            #         print(f"✅ Retry with k={k_retry} produced valid similarities")
            #     except Exception as e:
            #         print(f"Retry SVD with k={k_retry} failed: {e}")
            
            # Get feature names for interpretability
            feature_names = vectorizer.get_feature_names_out()
            
            # Interpret SVD dimensions
            dimension_labels = interpret_svd_dimensions(vt, feature_names)
            
            matching = []
            for i, (name, sim) in enumerate(zip(museum_names, similarities)):
                if name not in filtered_museums:
                    continue
                try:
                    doc_index = i - 1  # Adjust index for docs_vecs
                    if doc_index >= 0 and doc_index < len(docs_vecs_norm):
                        # Find top dimension for this document-query pair
                        top_dim_index, _ = get_dominant_dimension(query_vec_norm, docs_vecs_norm[doc_index])
                        dimension_name = dimension_labels[top_dim_index]
                        
                        # Get a representative review
                        review = get_representative_review(input_query,name, dataset)
                        
                        address = dataset[dataset['MuseumName'] == name]['Address'].values[0]
                        matching.append((name, address, float(sim), review, dimension_name))
                    else:
                        address = dataset[dataset['MuseumName'] == name]['Address'].values[0]
                        review = get_representative_review(input_query, name, dataset)
                        matching.append((name, address, float(sim), review, "Dimension not available"))
                except Exception as e:
                    print(f"Error getting data for museum {name}: {e}")
                    matching.append((name, "Address not found", float(sim), "No review available", "Unknown dimension"))

        except Exception as e:
            print(f"SVD calculation failed with error: {e}")
            query_vector = td_matrix[0:1]
            docs_vectors = td_matrix[1:]
            similarities = (query_vector @ docs_vectors.T).toarray().flatten()
            
            matching = []
            for i, (name, sim) in enumerate(zip(museum_names, similarities)):
                if filtered_museums is None or name not in filtered_museums:
                    continue
                try:
                    address = dataset[dataset['MuseumName'] == name]['Address'].values[0]
                    review = get_representative_review(input_query,name, dataset)
                    matching.append((name, address, float(sim), review, "SVD calculation failed"))
                except Exception as e:
                    print(f"Error getting data for museum {name}: {e}")
                    matching.append((name, "Address not found", float(sim), "No review available", "Unknown dimension"))

    matching.sort(key=lambda x: x[2], reverse=True)
    # make sure only positive cosine sim score results show up
    for i in range(len(matching)):
        t = matching[i]
        if t[2] <= 0.01:
            print(f"Low sim: {t[0]} -> {t[2]}")
            if t[2] > 0:
                matching[i] = (t[0],t[1],0.01,t[3],t[4])
    matching = [t for t in matching if t[2]>=0.01]

    # make sure no duplicate results show up
    final_matching = []
    seen = set()
    for tup in matching:
        if tup[0] not in seen:
            final_matching.append(tup)
            seen.add(tup[0])
    return final_matching