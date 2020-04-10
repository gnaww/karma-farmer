from sqlalchemy import Column, Integer, String
from . import db


class Word(db.Model):
    __tablename__ = "words"
    id = Column(Integer, primary_key=True)
    frequency = Column(Integer)
    netScore = Column(Integer)
    subreddit = Column(String)
    word = Column(String)

    def __init__(self, frequency=None, netScore=None, subreddit=None, word=None):
        self.frequency = frequency
        self.netScore = netScore
        self.subreddit = subreddit
        self.word = word
