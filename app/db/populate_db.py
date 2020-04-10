from nltk.tokenize import TweetTokenizer
from . import db, Word, SUBREDDITS_LIST
import numpy as np
import requests
import json

tweet_tokenizer = TweetTokenizer()


def fetch_data(subreddit):
    # Sort by score descending
    # Get top 50 results
    # Get posts after Wednesday, January 1, 2020 12:00:00 AM GMT
    res = requests.get(
        """https://api.pushshift.io/reddit/search/submission/?subreddit=%s&sort_type=score&sort=desc&size=50&after=1577836800&fields=score,selftext,title"""
        % (subreddit)
    )
    # Return value example : [{"score": 123, "selftext": "Body text", "title": "Title text"}]
    return res.json()["data"]


# TODO: better algorithm?
def process_data(data_arr):
    processed_data = np.zeros((0, 2), int)  # [frequency, score]
    index_counter = 0
    word_to_index = {}

    for data in data_arr:
        score = data["score"]
        # Make all text lowercase & account for empty fields
        body_text = data["selftext"].lower() if "selftext" in data else ""
        title_text = data["title"].lower() if "title" in data else ""

        # Tokenize texts
        tokens = tweet_tokenizer.tokenize(body_text + " " + title_text)

        # Process each token
        for token in tokens:
            if token in word_to_index:
                ind = word_to_index[token]
                processed_data[ind][0] += 1  # Update frequency
                processed_data[ind][1] += score  # Update score
            else:
                word_to_index[token] = index_counter
                processed_data = np.insert(
                    processed_data, index_counter, [1, score], axis=0
                )
                index_counter += 1

    return (word_to_index, processed_data)


def generate_db_objects(subreddit, word_to_index, processed_matrix):
    db_objects = []
    for word in word_to_index:
        ind = word_to_index[word]
        frequency = int(processed_matrix[ind][0])
        netScore = int(processed_matrix[ind][1])
        db_objects.append(Word(frequency, netScore, subreddit, word))
    return db_objects


def populate_db():
    # Reset database tables
    db.drop_all()
    db.create_all()

    # Process data for each subreddit
    for subreddit in SUBREDDITS_LIST:
        fetched_data = fetch_data(subreddit)
        word_to_index, processed_data = process_data(fetched_data)
        db_objects = generate_db_objects(subreddit, word_to_index, processed_data)
        db.session.bulk_save_objects(db_objects)

    # Persist changes to db
    db.session.commit()
