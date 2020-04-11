#!/usr/bin/python3

import csv
import matplotlib.pyplot as plt

def word_count_plot(data):
  words = {k: len(v) for k, v in data.items()}

  plt.hist(list(words.values()), 6)
  plt.xlabel('Number of Subreddits Word is Mentioned in')
  plt.ylabel('Number of Words in Bin')
  plt.title('Word Count Distribution Analysis in Various Subreddits')
  plt.axis([1, 6, 0, 4000])
  plt.grid(True)
  plt.show()

def word_freq_plot(data):
  plt.hist(list(data.values()), 100)
  plt.xlabel('Number of Occurrences of Word in Selected Subreddits')
  plt.ylabel('Number of Words in Bin')
  plt.title('Word Frequency in Various Subreddits')
  plt.axis([0, 100, 0, 4500])
  plt.grid(True)
  plt.show()
    

if __name__ == '__main__':
  with open('words.csv') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    next(csv_reader)
    subreddit_count = {}
    word_freq = {}
    for row in csv_reader:
      word, subreddit, frequency = row["word"], row["subreddit"], row["frequency"]

      if word in subreddit_count and subreddit not in subreddit_count[word]:
        subreddit_count[word].add(subreddit)
      elif word not in subreddit_count:
        subreddit_count[word] = {subreddit}

      if word in word_freq:
        word_freq[word] += int(frequency)
      elif word not in word_freq:
        word_freq[word] = int(frequency)
      
    word_count_plot(subreddit_count)
    word_freq_plot(word_freq)
