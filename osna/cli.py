# -*- coding: utf-8 -*-

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

from . import credentials_path, config

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
    Read all files in this directory and its subdirectories and print statistics.
    """
    print('reading from %s' % directory)
    # use glob to iterate all files matching desired pattern (e.g., .json files).
    # recursively search subdirectories.
    
    import pandas as pd
    # use the directory variable (passed into this function), 
    # rather than hard-coding this path. -awc
    #df=pd.read_csv('C:\\Users\\郑\\training_data\\data.csv\\data.csv')
    df=pd.read_csv(directory)
    
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
    print("The 50 most common words not hostile are %s"%words(b).most_common(50))
    print("The 50 most common words not hostile are %s"%words(c).most_common(50))


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
