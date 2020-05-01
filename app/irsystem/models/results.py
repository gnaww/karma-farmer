import math
import numpy as np
import string
from app.db import Data
from app.db import Metadata
from nltk.tokenize import TweetTokenizer
from spellchecker import SpellChecker


def get_data():
    return [data.serialize() for data in Data.query.all()]

def get_metadata():
    metadata = [data.serialize() for data in Metadata.query.all()]
    return {m["subreddit"]: (m["description"], m["subscribers"], m["posts"]) for m in metadata}

def get_good_types(posts):
    good_types = set()
    for post in posts:
        for word in post:
            good_types.add(word)
    good_types = np.array(list(good_types))
    good_types_lookup = {k: v for v, k in enumerate(good_types)}
    return good_types, good_types_lookup

def build_cooccurrence_mat(posts, good_types_lookup):
    tf_mat = np.zeros((len(posts), len(good_types_lookup)))
    for idx, post in enumerate(posts):
        for word in post:
            word_idx = good_types_lookup[word]
            tf_mat[idx, word_idx] += 1
    cooccurrence_mat = np.dot(tf_mat.transpose(), tf_mat)
    max_vals = np.amax(cooccurrence_mat, axis=0)
    cooccurrence_mat = np.divide(cooccurrence_mat, max_vals)
    return cooccurrence_mat

def get_subreddit_metadata(subreddit, metadata, query, idf):
    # TODO: get the words that have the highest cooccurrence with all of the words
    description = metadata[subreddit][0]
    subscribers = metadata[subreddit][1]
    posts = metadata[subreddit][2]

    query_idf = []
    for term in query:
        if term in idf:
            query_idf.append((term, idf[term]))
    query_idf = sorted(query_idf, key=lambda x: x[1], reverse=True)

    good_types, good_types_lookup = get_good_types(posts)
    cooccurrence_mat = build_cooccurrence_mat(posts, good_types_lookup)

    for term in query_idf:
        if term[0] in good_types_lookup.keys():
            highest_weighted_term_idx = good_types_lookup[term[0]]
            break
    # get the word with the highest index score and top 3 words
    highest_cooccur = cooccurrence_mat[highest_weighted_term_idx].argsort()[::-1]
    suggested_words = []
    i = 0
    while len(suggested_words) < 3 and i < len(highest_cooccur):
        word = good_types[highest_cooccur[i]]
        print(word)
        if word not in query:
            suggested_words.append(word)
        i += 1
    return description, subscribers, suggested_words

def build_inverted_index(data):
    id_to_subreddit = {}
    inverted_index = {}
    max_freq = 0
    max_score = 0
    for idx, entry in enumerate(data):
        subreddit = entry["subreddit"]
        id_to_subreddit[idx] = subreddit
        for word in entry["words"]:
            w = word["word"]
            if w in inverted_index:
                inverted_index[w].append((idx, word["frequency"], word["netScore"]))
            else:
                inverted_index[w] = [(idx, word["frequency"], word["netScore"])]
            if word["frequency"] > max_freq:
                max_freq = word["frequency"]
            if word["netScore"] > max_score:
                max_score = word["netScore"]
    for word in inverted_index:
        w = inverted_index[word]
        # TODO: shouldn't be dividing by max score because haven't filtered out words that are too frequent yet
        inverted_index[word] = [(i[0], i[1] / max_freq, i[2] / max_score) for i in w]
    return inverted_index, id_to_subreddit

def normalize_max(idf):
    idf_sum = max([i[1] for i in idf.items()]) or 1
    return dict(map(lambda x: (x[0], x[1] / idf_sum), idf.items()))

def normalize_avg(idf):
    idf_avg = np.mean([i[1] for i in idf.items()]) or 1
    return dict(map(lambda x: (x[0], x[1] / idf_avg), idf.items()))

def compute_idf(index, n_docs, min_df=2, max_df_ratio=0.90):
    total_karma = 0
    for term in index:
        n_term = len(index[term])
        if n_term >= min_df and n_term / n_docs <= max_df_ratio:
            total_karma += sum([term[2] for term in index[term]])

    idf = {}
    idf_score = {}
    for term in index:
        n_term = len(index[term])
        if n_term >= min_df and n_term / n_docs <= max_df_ratio:
            idf[term] = math.log(n_docs / (1 + n_term), 2)
            idf_score[term] = math.log(
                total_karma / (1 + sum([term[2] for term in index[term]])), 2
            )
    return normalize_avg(idf), normalize_avg(idf_score)

def compute_doc_norms(index, idf, n_docs):
    doc_norms_freq, doc_norms_score = np.zeros(n_docs), np.zeros(n_docs)
    for term in index:
        if term in idf:
            for doc in index[term]:
                doc_norms_freq[doc[0]] += (doc[1] * idf[term]) ** 2
                doc_norms_score[doc[0]] += (doc[2] * idf[term]) ** 2
    doc_norms_freq = np.sqrt(doc_norms_freq)
    doc_norms_score = np.sqrt(doc_norms_score)
    doc_norms_freq = doc_norms_freq / np.amax(doc_norms_freq)
    doc_norms_score = doc_norms_score / np.amax(doc_norms_score)
    return doc_norms_freq, doc_norms_score

def index_search(
    query,
    index,
    idf,
    idf_score,
    doc_norms_freq,
    doc_norms_score,
    tokenizer,
    id_to_subreddit,
    search_weight,
    score_weight,
    metadata
):
    results = []
    results_mat = np.zeros(len(doc_norms_freq))

    query = tokenizer.tokenize(query.lower())

    # Account for misspelled words
    spell = SpellChecker()
    misspelled = spell.unknown(query)
    word_correction = {}
    for word in misspelled:
        word_correction[word] = spell.correction(word)
    for ind, token in enumerate(query):
        if token in string.punctuation:  # don't autocorrect punctuation
            continue
        if token in word_correction:  # autocorrect
            query[ind] = word_correction[token]

    query_counts = [(t, query.count(t)) for t in set(query)]
    query_norm = 0

    for term in query_counts:
        if term[0] in idf:
            query_norm += (
                search_weight * (term[1] * idf[term[0]]) ** 2
                + score_weight * (term[1] * idf_score[term[0]]) ** 2
            )
            docs = index[term[0]]
            for doc in docs:
                results_mat[doc[0]] += (
                    search_weight * term[1] * doc[1] * idf[term[0]] ** 2
                    + score_weight * term[1] * doc[2] * idf_score[term[0]] ** 2
                )

    query_norm = math.sqrt(query_norm)

    for i, doc in enumerate(results_mat):
        den = query_norm * (
            search_weight * doc_norms_freq[i] + score_weight * doc_norms_score[i]
        )
        den = 1
        score = doc / den
        results.append(
            {"subreddit": id_to_subreddit[i], "score": score}
        )

    results = list(filter(lambda x: x["score"] > 0, sorted(results, key=lambda x: x["score"], reverse=True)))
    results = results[:5]
    for result in results:
        description, subscribers, suggested_words = get_subreddit_metadata(
                                                        result["subreddit"],
                                                        metadata,
                                                        set(query),
                                                        idf)
        result["description"] = description
        result["subscribers"] = subscribers
        result["suggested_words"] = suggested_words
    return results


def get_results(query, weight):
    data = get_data()
    metadata = get_metadata()
    n_subreddits = len(data)
    inv_idx, id_to_subreddit = build_inverted_index(data)
    idf, idf_score = compute_idf(inv_idx, n_subreddits)
    doc_norms_freq, doc_norms_score = compute_doc_norms(inv_idx, idf, n_subreddits)
    search_weight = int(weight) / 100
    score_weight = 1 - search_weight
    search = index_search(
        query,
        inv_idx,
        idf,
        idf_score,
        doc_norms_freq,
        doc_norms_score,
        TweetTokenizer(),
        id_to_subreddit,
        search_weight,
        score_weight,
        metadata
    )
    return search
