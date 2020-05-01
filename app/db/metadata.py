from sqlalchemy import Column, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY
from . import db
import json


class Metadata(db.Model):
    __tablename__ = "metadatas"
    id = Column(Integer, primary_key=True)
    subreddit = Column(String)
    description = Column(String)
    subscribers = Column(Integer)
    posts = Column(ARRAY(String))

    def __init__(self, subreddit=None, description="", subscribers=0, posts=None):
        self.subreddit = subreddit
        self.description = description
        self.subscribers = subscribers
        self.posts = posts

    def serialize(self):
        return {
            "subreddit": self.subreddit,
            "description": self.description,
            "subscribers": self.subscribers,
            "posts": [e.split(",") for e in self.posts],
        }
