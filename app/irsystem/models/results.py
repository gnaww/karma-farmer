import math
import numpy as np
from app.db import Data
from nltk.tokenize import TreebankWordTokenizer

def index_search(query, index, idf, doc_norms, tokenizer, id_to_subreddit):
    results = []
    results_mat = np.zeros(len(doc_norms))
    
    query = tokenizer.tokenize(query.lower())
    query_counts = [(t, query.count(t)) for t in set(query)]
    query_norm = 0
    
    for term in query_counts:
        if term[0] in idf:
            query_norm += (term[1]*idf[term[0]])**2
    query_norm = math.sqrt(query_norm)
    
    for term in query_counts:
        if term[0] in index:
            docs = index[term[0]]
            for doc in docs:
                results_mat[doc[0]] += term[1]*doc[1]*idf[term[0]]**2
                
    for i, doc in enumerate(results_mat):
        den = query_norm * doc_norms[i]
        den = 1 if den == 0 else den
        score = doc/den
        results.append((id_to_subreddit[i], score))
        
    results = sorted(results, key=lambda x:x[1], reverse=True)
    return results

def build_inverted_index(data):
    subreddit_to_id = {}
    id_to_subreddit = {}
    inverted_index = {}
    for idx, entry in enumerate(data):
        subreddit = entry['subreddit']
        subreddit_to_id[subreddit] = idx
        id_to_subreddit[idx] = subreddit
        for word in entry['words']:
            w = word['word']
            if w in inverted_index:
                inverted_index[w].append((idx, word['frequency'], word['netScore']))
            else:
                inverted_index[w] = [(idx, word['frequency'], word['netScore'])]
    return inverted_index, subreddit_to_id, id_to_subreddit

# don't have enough docs yet to use min/max df
# def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
def compute_idf(inv_idx, n_docs):
    idf = {}
    for term in inv_idx:
        n_term = len(inv_idx[term])
        # if n_term >= min_df and n_term/n_docs <= max_df_ratio:
        idf[term] = math.log(n_docs/(1+n_term), 2)
    return idf

def compute_doc_norms(index, idf, n_docs):   
    doc_norms = np.zeros(n_docs)
    for term in index:
        # don't need to check this cond unless we filter min/max df
        # if term in idf:
        for doc in index[term]:
            doc_norms[doc[0]] += (doc[1] * idf[term])**2
    doc_norms = np.sqrt(doc_norms)
    return doc_norms

def get_data():
    return [data.serialize() for data in Data.query.all()]

def get_results(query):
    data = get_data()
    n_subreddits = len(data)
    inv_idx, subreddit_to_id, id_to_subreddit = build_inverted_index(data)
    idf = compute_idf(inv_idx, n_subreddits)
    doc_norms = compute_doc_norms(inv_idx, idf, n_subreddits)
    search = index_search(query, inv_idx, idf, doc_norms, TreebankWordTokenizer(), id_to_subreddit)
    return search