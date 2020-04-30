import requests
import json
import time
import re

def fetch_subreddits(link_id):
    res = requests.get(
        """https://api.pushshift.io/reddit/comment/search/?link_id=%s&score=>15&size=5000"""
        % link_id
    )
    subreddit_set = set()

    for comment in res.json()['data']:
        potential_subreddits = re.split("\/?r\/", comment['body'])
        # length is more than 1 if there was an r/
        if len(potential_subreddits) > 1:
            # if first element isn't '', then the comment didn't start out with r/
            if potential_subreddits[0] != '':
                del potential_subreddits[0]
            # print(comment['body'])
            # print(potential_subreddits)
            for subreddit in potential_subreddits:
                subreddit = subreddit.lower()
                # remove any non alphanumeric character after the extracted subreddit
                subreddit = re.split("[^a-zA-Z\d]", subreddit)[0]
                if subreddit != '':
                    idx = re.search("\s", subreddit)
                    if idx is None:
                        subreddit_set.add(subreddit)
                    else:
                        idx = idx.start()
                        subreddit_set.add(subreddit[:idx])
    # print(subreddit_set)
    return subreddit_set


if __name__ == "__main__":
    prev_list = [
        "conspiracy",
        "movies",
        "music",
        "todayilearned",
        "askreddit",
        "unpopularopinion",
        "worldnews",
        "news",
        "politics",
        "showerthoughts",
        "teenagers",
        "history",
        "changemyview",
        "explainlikeimfive",
        "books",
        "games",
        "technology",
        "travel",
        "offmychest",
        "cars",
        "worldpolitics",
        "askscience",
        "casualconversation",
        "tipofmytongue",
        "iama",
        "hockey",
        "philosophy",
        "business",
        "nostupidquestions",
        "outoftheloop",
        "askhistorians",
        "science",
        "jokes",
        "television",
        "programming",
        "boardgames",
        "askmen",
        "askwomen",
        "crazyideas",
        "hiphopheads",
        "apple",
        "tooafraidtoask",
        "lifeprotips",
        "nottheonion",
        "space",
        "gadgets",
        "upliftingnews",
        "tifu",
        "futurology",
        "writingprompts",
        "nosleep",
        "twoxchromosomes",
        "personalfinance",
        "dadjokes",
        "iphone",
        "android",
        "buildapc",
        "confession",
        "fitness",
        "leagueoflegends",
        "relationships",
        "atheism",
        "relationship_advice",
        "gameofthrones",
        "youshouldknow",
        "parenting"
    ]
    links = ["9l00b1", "cigqig", "aqynop", "g3a9tc", "98dms8", "6jc4n0", "fg48c9"]
    subreddit_list = set()
    for link in links:
        subreddits = fetch_subreddits(link)
        subreddit_list = subreddit_list.union(subreddits)
    subreddit_list = subreddit_list.union(set(prev_list))
    subreddit_list = list(subreddit_list)
    print(len(subreddit_list))
    print(subreddit_list)
    json_dict = { "subreddits": subreddit_list }

    with open('subreddits.json', 'w') as f:
        json.dump(json_dict, f, ensure_ascii=False, indent=4)