import math
import numpy as np
from app.db import Data
from nltk.tokenize import TreebankWordTokenizer

def get_data():
    return [data.serialize() for data in Data.query.all()]

def build_inverted_index(data):
    id_to_subreddit = {}
    inverted_index = {}
    for idx, entry in enumerate(data):
        subreddit = entry['subreddit']
        id_to_subreddit[idx] = subreddit
        for word in entry['words']:
            w = word['word']
            if w in inverted_index:
                inverted_index[w].append((idx, word['frequency'], word['netScore']))
            else:
                inverted_index[w] = [(idx, word['frequency'], word['netScore'])]
    return inverted_index, id_to_subreddit

# don't have enough docs yet to use min/max df
# def compute_idf(inv_idx, n_docs, min_df=10, max_df_ratio=0.95):
def compute_idf(index, n_docs):
    idf = {}
    for term in index:
        n_term = len(index[term])
        # if n_term >= min_df and n_term/n_docs <= max_df_ratio:
        idf[term] = math.log(n_docs/(1+n_term), 2)
    return idf

def compute_doc_norms(index, idf, n_docs):   
    doc_norms_freq, doc_norms_score = np.zeros(n_docs), np.zeros(n_docs)
    for term in index:
        # don't need to check this condition unless we filter min/max df
        # if term in idf:
        for doc in index[term]:
            doc_norms_freq[doc[0]] += (doc[1] * idf[term])**2
            doc_norms_score[doc[0]] += (doc[2] * idf[term])**2
    doc_norms_freq = np.sqrt(doc_norms_freq)
    doc_norms_score = np.sqrt(doc_norms_score)
    print(doc_norms_freq)
    print(doc_norms_score)
    return doc_norms_freq, doc_norms_score

def index_search(query, index, idf, doc_norms_freq, doc_norms_score, tokenizer, id_to_subreddit, search_weight=0.95, score_weight=0.05):
    results = []
    results_mat = np.zeros(len(doc_norms_freq))
    
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
                results_mat[doc[0]] += search_weight*term[1]*doc[1]*idf[term[0]]**2 + score_weight*term[1]*doc[2]*idf[term[0]]**2

    for i, doc in enumerate(results_mat):
        den = query_norm * doc_norms_freq[i]
        den = 1 if den == 0 else den
        score = doc/den
        results.append({'subreddit': id_to_subreddit[i], 'score': score, 'suggested_words': []})
        
    results = sorted(results, key=lambda x:x['score'], reverse=True)
    return results

def get_results(query):
    data = get_data()
    n_subreddits = len(data)
    inv_idx, id_to_subreddit = build_inverted_index(data)
    idf = compute_idf(inv_idx, n_subreddits)
    doc_norms_freq, doc_norms_score = compute_doc_norms(inv_idx, idf, n_subreddits)
    search = index_search(query, inv_idx, idf, doc_norms_freq, doc_norms_score, TreebankWordTokenizer(), id_to_subreddit)
    return search