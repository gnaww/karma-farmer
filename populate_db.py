from nltk.tokenize import TweetTokenizer
# from . import db, Data, SUBREDDITS_LIST
from app.db import db
from app.db import Data
from app.db import SUBREDDITS_LIST
import numpy as np
import requests
import json
import time

tweet_tokenizer = TweetTokenizer()


def fetch_data(subreddit):
    # Sort by score descending
    # Get top 50 results from past 30 days
    res = requests.get(
        """https://api.pushshift.io/reddit/search/submission/?subreddit=%s&sort_type=score&sort=desc&size=50&after=30d&fields=score,selftext,title&over_18=false"""
        % (subreddit)
    )

    print(subreddit, res.status_code)
    # request failed because of too many requests in too little time
    if res.status_code == 429:
        while res.status_code != 200:
            time.sleep(0.5)
            res = requests.get(
                """https://api.pushshift.io/reddit/search/submission/?subreddit=%s&sort_type=score&sort=desc&size=50&after=30d&fields=score,selftext,title&over_18=false"""
                % (subreddit)
            )
            print(subreddit, res.status_code)
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


def generate_strings(word_to_index, processed_matrix):
    str_arr = []
    for word in word_to_index:
        ind = word_to_index[word]
        frequency = int(processed_matrix[ind][0])
        netScore = int(processed_matrix[ind][1])
        obj = json.dumps(
            {"word": word, "frequency": frequency, "netScore": netScore}
        )  # dictionary to str
        str_arr.append(obj)
    return str_arr


def populate_db():
    # Reset database tables
    db.drop_all()
    db.create_all()

    # Process data for each subreddit
    for subreddit in SUBREDDITS_LIST:
        fetched_data = fetch_data(subreddit)
        word_to_index, processed_data = process_data(fetched_data)
        str_arr = generate_strings(word_to_index, processed_data)
        db.session.add(Data(subreddit, str_arr))

    # Persist changes to db
    db.session.commit()

if __name__ == "__main__":
    populate_db()