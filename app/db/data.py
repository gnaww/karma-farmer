from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY
from . import db
import json


class Data(db.Model):
    __tablename__ = "datas"
    id = Column(Integer, primary_key=True)
    subreddit = Column(String)
    description = Column(String)
    subscribers = Column(Integer)
    words = Column(ARRAY(String))

    def __init__(self, subreddit=None, description="", subscribers=0, words=None):
        self.subreddit = subreddit
        self.description = description
        self.subscribers = subscribers
        self.words = words

    def serialize(self):
        return {
            "subreddit": self.subreddit,
            "description": self.description,
            "subscribers": self.subscribers,
            "words": [json.loads(e) for e in self.words],
        }
