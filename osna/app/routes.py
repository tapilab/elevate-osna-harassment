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
    sort_coef=[]
    coef = [-clf.coef_[0], clf.coef_[0]]
    features = np.array(vec.get_feature_names())
    for i in range(0,len(coef[0])):
        sort_coef.append([coef[0][i],features[i]])
    myList = sorted(sort_coef, key=lambda x: x[0])
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
            p='[probability='+'%.2f'%(proba[i,y[i]])+'] '
            coef_text=''
            a = [0]
            for j in np.argsort(coef[0][X[i].nonzero()[1]])[::-1][:3]:#start step stop
                idx = X[i].nonzero()[1][j]
                a[0] += 1
                coef_text=coef_text+'//    '+'%d'%a[0]+'.'+str(features[idx])+': '+'%.2f'%(coef[0][idx])+'  '
            # ans.append(flag+p+tweets[i]+coef_text)
            ans.append(flag + p)
            ans.append(tweets[i])
            ans.append(coef_text)
            ans.append("\n")
        #Tweets = a list of dict
        return render_template('myform.html', title='', form=form, tweets=ans)
        #return redirect('/index')
    return render_template('myform.html', title='', form=form)
