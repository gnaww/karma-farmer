from nltk.tokenize import TweetTokenizer
from html2text import html2text
from app.db import db, Data, Metadata, COMMON_WORDS
import numpy as np
import requests
import json
import time
import string

tweet_tokenizer = TweetTokenizer()


def get_subreddit_info(subreddit):
    while True:  # Fail-safe logic for 429 error code (too many requests)
        res = requests.get(
            "https://www.reddit.com/r/%s/about.json" % (subreddit),
            allow_redirects=False,
        )
        print("[Getting subreddit info]", subreddit, res.status_code)

        if res.status_code == 200:  # Successful request : Valid subreddit
            print("Valid subreddit")
            data = res.json()["data"]
            if data["over18"]:  # Inappropriate subreddit
                print("Inappropriate subreddit")
                return (False, None, None)

            # Return information about subreddit
            description = data["public_description"]
            description = html2text(description).strip()
            subscribers = data["subscribers"]
            return (True, description, subscribers)
            break
        elif res.status_code != 429:  # Successful request : Invalid subreddit
            print("Invalid subreddit")
            return (False, None, None)
            break

        time.sleep(0.5)  # Unsuccessful request: Try again with delay


def fetch_data(subreddit):
    valid_posts = []
    valid_posts_counter = 0
    after = ""  # request parameter for pagination view

    while True:  # Fail-safe logic for 429 error code (too many requests)
        # Get results from subreddit ranked by scores from last 30 days
        res = requests.get(
            "https://www.reddit.com/r/%s/top.json?t=month&limit=100&after=%s"
            % (subreddit, after)
        )
        print("[Getting posts]", subreddit, res.status_code)
        print("Valid posts :", valid_posts_counter)

        if res.status_code == 200:  # Successful request
            data = res.json()["data"]
            after = data["after"]  # for next request
            children = data["children"]

            for post in children:  # loop through posts
                post_data = post["data"]
                # Skip post : inappropriate post OR empty post body
                if post_data["over_18"] or post_data["selftext"] == "":
                    continue

                # Process post
                valid_posts.append(
                    {
                        "score": post_data["score"],
                        "selftext": post_data["selftext"],
                        "title": post_data["title"],
                    }
                )
                valid_posts_counter += 1

                # If reached 500 text-based posts OR no more results to fetch, return list.
                # If not, do another iteration of while loop to get next batch of results to process
                if valid_posts_counter == 500:
                    print("[DONE] Valid posts :", valid_posts_counter)
                    # Return value example : [{"score": 123, "selftext": "Body text", "title": "Title text"}]
                    return valid_posts
                    break

            if len(children) == 0 or after is None:  # No more posts to process
                return valid_posts

        time.sleep(0.5)  # Unsuccessful request: Try again with delay


def process_data(data_arr):
    processed_data = np.zeros((0, 2), int)  # [frequency, score]
    index_counter = 0
    word_to_index = {}
    posts_tokenized = []

    # For each post
    for data in data_arr:
        score = data["score"]
        # Make all text lowercase & account for empty fields
        body_text = data["selftext"].lower() if "selftext" in data else ""
        title_text = data["title"].lower() if "title" in data else ""

        # Tokenize texts
        tokens = tweet_tokenizer.tokenize(body_text + " " + title_text)

        # Array of valid tokens for post
        post_tokens = []

        # Process each token
        for token in tokens:
            # Skip punctuations or common words or single letters or token is a link
            if (
                token in string.punctuation
                or token in COMMON_WORDS
                or len(token) == 1
                or "http" in token
            ):
                continue

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

            # Update tokens for post
            post_tokens.append(token)

        posts_tokenized.append(",".join(post_tokens))  # Stringify and append to list

    return (word_to_index, processed_data, posts_tokenized)


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

    # Get list of subreddits
    with open("app/db/subreddits.json") as f:
        data = json.load(f)
        SUBREDDITS_LIST = data["subreddits"]
        f.close()

    # Process data for each subreddit
    for subreddit in SUBREDDITS_LIST:
        sr_valid, sr_description, sr_subscribers = get_subreddit_info(subreddit)
        # If invalid/inappropriate subreddit, skip processing information
        if not sr_valid:
            continue

        fetched_data = fetch_data(subreddit)
        word_to_index, processed_data, post_arr = process_data(fetched_data)
        word_arr = generate_strings(word_to_index, processed_data)
        db.session.add(Metadata(subreddit, sr_description, sr_subscribers, post_arr))
        db.session.add(Data(subreddit, word_arr))
        db.session.commit()


if __name__ == "__main__":
    populate_db()
