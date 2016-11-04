# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 20:14:28 2016

@author: Xin
"""

import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
import numpy as np

## Skytrax data
airline = pd.read_csv('airline.csv')
airport = pd.read_csv('airport.csv')
seat = pd.read_csv('seat.csv')
lounge = pd.read_csv('lounge.csv')

# Extract reviews related to airlines from airline, seat, and lounge datasets
# select only US major airlines for this analysis
names = ['american-airlines', 'alaska-airlines','jetblue-airways',
'continental-airlines','delta-air-lines','eastern-airways','frontier-airlines',
'hawaiian-airlines','spirit-airlines','comair', 'skywest-airlines',
'united-airlines','us-airways','virgin-america','southwest-airlines']
airline_us = airline.loc[airline['airline_name'].isin(names)]
seat_us = seat.loc[seat['airline_name'].isin(names)]
lounge_us = lounge.loc[lounge['airline_name'].isin(names)]

# select airline_name, reviews content and overall_rating for text mining
reviews_al = airline_us[['airline_name', 'content', 'overall_rating']]
reviews_st = seat_us[['airline_name', 'content', 'overall_rating']]
reviews_lg = lounge_us[['airline_name', 'content', 'overall_rating']]

# convert overall rating from numeric to categorical rating: Positive, Neutral, Negative
reviews_al = reviews_al.dropna(subset=['overall_rating'])
categories = {1.0:'Negative', 2.0:'Negative',3.0:'Negative',4.0:'Neutral',5.0:'Neutral',6.0:'Neutral',7.0:'Neutral',8.0:'Positive',9.0:'Positive',10.0:'Positive'}
reviews_al['rating'] = reviews_al['overall_rating'].map(lambda x: categories[x])

# Text feature extraction using BOW model
def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = TextBlob(message).words
    return [word.lemma for word in words]

# Convert a collection of reviews to a matrix of token counts
BOW_transform = CountVectorizer(analyzer=split_into_lemmas).fit(reviews_al['content'])

BOW = BOW_transform.transform(reviews_al['content'])

print 'Shape of the large transformed matrix is:', BOW.shape, 'i.e', BOW.shape[0],'rows that corresponds to the number of reviews and', BOW.shape[1], 'columns for each of the review that corresponds to the total number of unique words in the dataset'

# Term weighting and normalization using the TfidfTransformer of the Scikit-learn
# Learn the idf vector and then transform BOW count matrix to tfidf
tfidf_BOW = TfidfTransformer().fit(BOW)
converted_reviews = tfidf_BOW.transform(BOW)


# fit random forest model
target = np.array(reviews_al['rating'].tolist())
features = converted_reviews
rfc = RandomForestClassifier(n_estimators=500, oob_score=True)
model = rfc.fit(features, target)
importances = rfc.feature_importances_
indices = np.argsort(importances)[-50:]
featureword = []
for i in indices:
    featureword.append(BOW_transform.get_feature_names()[i])
print featureword

# food, seat, snack, crew, cleanness, delay, comfort