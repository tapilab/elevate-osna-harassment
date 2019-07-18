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

    SumList = []  # the word list which only appear once in all text
    accountAll = 0
    for message in df['text']:
        message = simple_tokenizer(message)
        accountAll += len(message)
        SumList.extend(message)
    counter = dict(Counter(SumList))
    count = 0
    for word in counter:
    		if counter[word] > 1:
    			count += counter[word]
    print(count)
    print('reading from %s' % directory)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
