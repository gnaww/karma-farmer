from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY
from . import db
import json


class Data(db.Model):
    __tablename__ = "datas"
    id = Column(Integer, primary_key=True)
    subreddit = Column(String)
    words = Column(ARRAY(String))

    def __init__(self, subreddit=None, words=None):
        self.words = words
        self.subreddit = subreddit

    def serialize(self):
        return {
            "words": [json.loads(e) for e in self.words],
            "subreddit": self.subreddit,
        }
