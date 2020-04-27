import math
import numpy as np
from app.db import Data
from nltk.tokenize import TreebankWordTokenizer

def get_data():
    return [data.serialize() for data in Data.query.all()]

def build_inverted_index(data):
    id_to_subreddit = {}
    inverted_index = {}
    max_freq = 0
    max_score = 0
    for idx, entry in enumerate(data):
        subreddit = entry['subreddit']
        id_to_subreddit[idx] = subreddit
        for word in entry['words']:
            w = word['word']
            if w in inverted_index:
                inverted_index[w].append((idx, word['frequency'], word['netScore']))
            else:
                inverted_index[w] = [(idx, word['frequency'], word['netScore'])]
            if word['frequency'] > max_freq:
                max_freq = word['frequency']
            if word['netScore'] > max_score:
                max_score = word['netScore']
    for word in inverted_index:
        w = inverted_index[word]
        inverted_index[word] = [(i[0], i[1]/max_freq, i[2]/max_score) for i in w]
    # somehow normalizing the inverted index has messed up results a lot
    return inverted_index, id_to_subreddit

def normalize(idf):
    idf_sum = max([i[1] for i in idf.items()])
    return dict(map(lambda x: (x[0], x[1]/idf_sum), idf.items()))

def compute_idf(index, n_docs, min_df=2, max_df_ratio=0.90):
    total_karma = 0
    for term in index:
        n_term = len(index[term])
        if n_term >= min_df and n_term/n_docs <= max_df_ratio:
            total_karma += sum([term[2] for term in index[term]])

    idf = {}
    idf_score = {}
    for term in index:
        n_term = len(index[term])
        if n_term >= min_df and n_term/n_docs <= max_df_ratio:
            idf[term] = math.log(n_docs/(1+n_term), 2)
            # not sure if total karma is the best thing to divide
            idf_score[term] = math.log(total_karma/(1+sum([term[2] for term in index[term]])), 2)
    return normalize(idf), normalize(idf_score)

def compute_doc_norms(index, idf, n_docs):   
    doc_norms_freq, doc_norms_score = np.zeros(n_docs), np.zeros(n_docs)
    for term in index:
        if term in idf:
            for doc in index[term]:
                doc_norms_freq[doc[0]] += (doc[1] * idf[term])**2
                doc_norms_score[doc[0]] += (doc[2] * idf[term])**2
    doc_norms_freq = np.sqrt(doc_norms_freq)
    doc_norms_score = np.sqrt(doc_norms_score)
    doc_norms_freq = doc_norms_freq/np.amax(doc_norms_freq)
    doc_norms_score = doc_norms_score/np.amax(doc_norms_score)
    return doc_norms_freq, doc_norms_score

def index_search(query, index, idf, idf_score, doc_norms_freq, doc_norms_score, tokenizer, id_to_subreddit, search_weight, score_weight):
    results = []
    results_mat = np.zeros(len(doc_norms_freq))
    
    query = tokenizer.tokenize(query.lower())
    query_counts = [(t, query.count(t)) for t in set(query)]
    query_norm = 0
    
    for term in query_counts:
        if term[0] in idf:
            query_norm += search_weight*(term[1]*idf[term[0]])**2 + score_weight*(idf_score[term[0]])**2
            docs = index[term[0]]
            for doc in docs:
                results_mat[doc[0]] += search_weight*term[1]*doc[1]*idf[term[0]]**2 + score_weight*doc[2]*idf_score[term[0]]**2

    query_norm = math.sqrt(query_norm)

    for i, doc in enumerate(results_mat):
        den = query_norm * (search_weight*doc_norms_freq[i] + score_weight*doc_norms_score[i])
        den = 1 if den == 0 else den
        score = doc/den
        results.append({'subreddit': id_to_subreddit[i], 'score': score, 'suggested_words': []})

    results = sorted(results, key=lambda x:x['score'], reverse=True)
    return results

def get_results(query, weight):
    data = get_data()
    n_subreddits = len(data)
    inv_idx, id_to_subreddit = build_inverted_index(data)
    idf, idf_score = compute_idf(inv_idx, n_subreddits)
    doc_norms_freq, doc_norms_score = compute_doc_norms(inv_idx, idf, n_subreddits)
    search_weight = int(weight)/100
    score_weight = 1 - search_weight
    search = index_search(query, inv_idx, idf, idf_score, doc_norms_freq, doc_norms_score, TreebankWordTokenizer(), id_to_subreddit, search_weight, score_weight)
    return search