# -*- coding: utf-8 -*-

"""Console script for elevate_osna."""
import click
import os  # operation system
import glob
import pickle
import sys
import numpy as np# -*- coding: utf-8 -*-

"""Console script for elevate_osna."""
import click
import os  # operation system
import glob
import sys
import numpy as np
import pandas as pd
import scipy
import json
import gzip
from collections import Counter

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, classification_report

from . import credentials_path, clf_path


# from . import credentials_path, config

@click.group()
def main(args=None):
    """Console script for osna."""
    return 0


@main.command('web')
@click.option('-t', '--twitter-credentials', required=False, type=click.Path(exists=True), show_default=True,
              default=credentials_path, help='a json file of twitter tokens')
@click.option('-p', '--port', required=False, default=5000, show_default=True, help='port of web server')
def web(twitter_credentials, port):
    from .app import app  # .app是访问同根的文件夹
    app.run(host='127.0.0.1', debug=True, port=port)

@main.command('stats')
@click.argument('directory', type=click.Path(exists=True))
def stats(directory):
    """
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
    Read all files in this directory and its subdirectories and print statistics.
    """
    df = pd.read_csv(directory)
    # get Unique username number
    User = []
    UniUser = 0
    for user in df['id']:
        # print(user)
        User.append(user)
        UniUser += 1
        for u in User[:len(User) - 1]:
            if (u == user):
                del User[-1]
                UniUser -= 1
                break

    # use glob to iterate all files matching desired pattern (e.g., .json files).
    # recursively search subdirectories.

    # ,text,target,sender,link,hostile,If hostile is it directed ?,group,id
    # the structure of json

    class solution:
        def FindNumsAppearNotOnce(self, array):
            # find element only appear once in this specific text (not in all)
            array.sort()
            li = []
            y = list(array)
            for i in range(len(y)):
                if array.count(y[i]) > 1:
                    li.append(y[i])
            print(li)
            return li

    def simple_tokenizer(s):
        return s.split()


    # use glob to iterate all files matching desired pattern (e.g., .json files).
    # recursively search subdirectories.
    
    unique_user=set(df['sender'])
    n1=len(unique_user)
    print("The number of unique users is %d"%n1)
    unique_message=set(df['text'])
    n2=len(unique_message)
    print("The number of unique messages is %d"%n2)
    

    
    unique_user=set(df['sender'])
    n1=len(unique_user)
    print("The number of unique users is %d"%n1)
    
    unique_message=set(df['text'])
    n2=len(unique_message)
    print("The number of unique messages is %d"%n2)
    

    df1=df[df['hostile']== 0]
    n3=len(set(df1['sender']))
    n4=len(set(df1['text']))
    print("The number of unique users not hostile is %d"%n3)
    print("The number of unique messages not hostile is %d"%n4)
    df2=df[df['hostile']== 1]
    n5=len(set(df2['sender']))
    n6=len(set(df2['text']))
    print("The number of unique users hostile is %d"%n5)
    print("The number of unique messages hostile is %d"%n6)

    
    import re
    def tweet_tokenizer(s):
        s = re.sub(r'#(\S+)', r'HASHTAG_\1', s)
        s = re.sub(r'@(\S+)', r'MENTION_\1', s)
        s = re.sub(r'http\S+', 'THIS_IS_A_URL', s)
        return re.sub('\W+', ' ', s.lower()).split()
    a = []
    for tweet in df['text']:
        a=a+tweet_tokenizer(tweet)


    SumList = []  # the word list which only appear once in all text
    accountAll = 0
    for message in df['text']:
        message = simple_tokenizer(message)
        accountAll += len(message)
        SumList.extend(message)
    counter = dict(Counter(SumList))
    count = 0
    for word in counter:
            if counter[word] >= 1:
                count += counter[word]
    print(count)
    # a = []
    # for tweet in df['text']:
    #     a=a+tweet_tokenizer(tweet)

    a = [t for tweet in df['text'] for t in tweet_tokenizer(tweet)]
    # a = []
    # for tweet in df['text']:
    #     for t in tweet:
    #         a.append(t)


    n7=len(set(a))
    n8=len(a)
    print("The number of unique words is %d"%n7)
    print("The number of tokens is %d"%n8)

    from collections import Counter
    def words(dicts):
        counts = Counter() # handy object: dict from object -> int
        counts.update(dicts)
        return counts
    counts = words(a)
    print("The 50 most common words are %s"%counts.most_common(50))
    
    b = []
    for tweet in df1['text']:
        b=b+tweet_tokenizer(tweet)
    c = []
    for tweet in df2['text']:
        c=c+tweet_tokenizer(tweet)

    
    from collections import Counter

    counts = words(a)
    print("The 50 most common words are %s"%counts.most_common(50))
    
    b = []
    for tweet in df1['text']:
        b=b+tweet_tokenizer(tweet)
    c = []
    for tweet in df2['text']:
        c=c+tweet_tokenizer(tweet)

    print("The 50 most common words not hostile are %s"%words(b).most_common(50))
    print("The 50 most common words not hostile are %s"%words(c).most_common(50))
    """
    Read all files in this directory and its subdirectories and print statistics.
    print('reading from %s' % directory)
    # use glob to iterate all files matching desired pattern (e.g., .json files).
    # recursively search subdirectories.
    """

@main.command('train')
@click.argument('directory', type=click.Path(exists=True))
def train(directory):
    """
    Train a classifier and save it.
    """
    print('reading from %s' % directory)
    # (1) Read the data...
    #
    # (2) Create classifier and vectorizer.
    clf = LogisticRegression() # set best parameters 
    vec = CountVectorizer()    # set best parameters
    
    # (3) do cross-validation and print out validation metrics
    # (classification_report)

    # (4) Finally, train on ALL data one final time and
    # train...
    # save the classifier
    pickle.dump((clf, vec), open(clf_path, 'wb'))


def make_features(df):
    ## Add your code to create features.
    pass

if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover


# from . import credentials_path, config

