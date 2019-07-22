from flask import render_template, flash, redirect, session
from . import app
from .forms import MyForm
#from ..mytwitter import Twitter
#from ..u import get_twitter_data, N_TWEETS
from .. import credentials_path
from TwitterAPI import TwitterAPI 
import json
import sys
from osna.mytwitter import Twitter 


#twapi = Twitter(credentials_path)

@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index():
	form = MyForm()
	result = None
	if form.validate_on_submit():
		input_field = form.input_field.data
		Output = 'Invalid'
		if(input_field=='get_tweet'):
    			Output=text
		flash(input_field)
		return render_template('myform.html', title='', form=form, tweets=Output)
	#return redirect('/index')
	return render_template('myform.html', title='', form=form)

twitter = Twitter.get_twitter('credentials.json')
print('Established Twitter connection.')
time=Twitter.robust_request(twitter,'statuses/user_timeline',
                          {'screen_name': '@realDonaldTrump',
                           'count':200} )
time2=Twitter.robust_request(twitter,'statuses/user_timeline',
							{'screen_name': '@pixy_qi',
							'count':200} )							



text=('\t'.join(t['text'] for t in time))
text2=('\t'.join(t['text'] for t in time2))
text+=text2

print('got %d tweets for user ' % (len(text)))