from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm
from ..mytwitter import Twitter
#from ..u import get_twitter_data, N_TWEETS
from .. import credentials_path, clf_path
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

import pickle
import sys
import json
import numpy as np
# import tensorflow as tf
from TwitterAPI import TwitterAPI

#twapi = Twitter(credentials_path)

clf, vec = pickle.load(open(clf_path, 'rb'))

print('read clf %s' % str(clf))
print('read vec %s' % str(vec))

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
    form = MyForm()
    result = None
    if form.validate_on_submit():
        input_field = form.input_field.data
        flash(input_field)
        t = Twitter(credentials_path)

        all_tweets = t._get_tweets('screen_name',input_field,limit=200)
        tweets = [words['full_text'] for words in all_tweets]

        X = vec.transform(text for text in tweets)
        y=clf.predict(X)
        proba=clf.predict_proba(X)

        ans=[]
        count_vec = CountVectorizer(min_df=1)
        for i in range(0,len(tweets)):
            flag='[hostile] '
            if y[i]==0:
                flag='[non-hostile] '
            p='[probability='+str(proba[i,y[i]])+'] '
            ans.append(flag+p+tweets[i]+tweets[i])

        #Tweets = a list of dict
        return render_template('myform.html', title='', form=form, tweets=ans)

        #return redirect('/index')
    return render_template('myform.html', title='', form=form)
