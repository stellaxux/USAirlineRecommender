# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 13:56:13 2016

@author: Xin
"""

import pandas as pd
from textblob import TextBlob
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
import math

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
categories = {1.0:'Negative', 2.0:'Negative',3.0:'Negative',4.0:'Negative',5.0:'Negative',6.0:'Negative',7.0:'Positive',8.0:'Positive',9.0:'Positive',10.0:'Positive'}
reviews_al['rating'] = reviews_al['overall_rating'].map(lambda x: categories[x])

# positive reviews 1762
posRev = reviews_al[reviews_al['rating']=='Positive']
posRev = posRev['content']

# negative reviews 2924
negRev = reviews_al[reviews_al['rating']=='Negative']
negRev = negRev['content']

stops = set(stopwords.words('english'))

posfeatures = []
for review in posRev:
    review = unicode(review, 'utf8').lower()
    words = TextBlob(review).words
    words_lemma = [word.lemma for word in words]
    words_filtered = [word for word in words_lemma if len(word)>=3 and word not in stops]
    posfeatures.append((words_filtered, 'positive'))
    
negfeatures = []
for review in negRev:
    review = unicode(review, 'utf8').lower()
    words = TextBlob(review).words
    words_lemma = [word.lemma for word in words]
    words_filtered = [word for word in words_lemma if len(word)>=3 and word not in stops]
    negfeatures.append((words_filtered, 'negative'))

#selects 3/4 of the features to be used for training and 1/4 to be used for testing
posCutoff = int(math.floor(len(posfeatures)*3/4))
negCutoff = int(math.floor(len(negfeatures)*3/4))
trainFeatures = posfeatures[:posCutoff] + negfeatures[:negCutoff]
testFeatures = posfeatures[posCutoff:] + negfeatures[negCutoff:]

def get_words_in_reviews(reviews):
    all_words = []
    for (words, sentiment) in reviews:
      all_words.extend(words)
    return all_words
    
def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features
    
def extract_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

word_features = get_word_features(get_words_in_reviews(trainFeatures))
training_set = nltk.classify.apply_features(extract_features, trainFeatures)
testing_set = nltk.classify.apply_features(extract_features, testFeatures)

#train the naive bayes classifier
classifier = NaiveBayesClassifier.train(training_set)
print classifier.show_most_informative_features(32)
print 'accuracy:', nltk.classify.util.accuracy(classifier, testing_set)

#Most Informative Features
#           contains(anc) = True           positi : negati =     26.0 : 1.0
#           contains(rep) = True           negati : positi =     20.3 : 1.0
#       contains(pleased) = True           positi : negati =     17.8 : 1.0
#       contains(refused) = True           negati : positi =     15.5 : 1.0
#      contains(lie-flat) = True           positi : negati =     14.9 : 1.0
#         contains(speak) = True           negati : positi =     13.6 : 1.0
#  contains(notification) = True           negati : positi =     13.1 : 1.0
#   contains(explanation) = True           negati : positi =     12.8 : 1.0
#    contains(impressive) = True           positi : negati =     12.7 : 1.0
#          contains(a321) = True           positi : negati =     12.7 : 1.0
#         contains(ample) = True           positi : negati =     12.7 : 1.0
#          contains(mint) = True           positi : negati =     12.7 : 1.0
#        contains(runway) = True           negati : positi =     11.9 : 1.0
#     contains(impressed) = True           positi : negati =     11.8 : 1.0
#       contains(relaxed) = True           positi : negati =     11.6 : 1.0
#     contains(announced) = True           negati : positi =     11.3 : 1.0
#     contains(excellent) = True           positi : negati =     11.2 : 1.0
#   contains(frustrating) = True           negati : positi =     11.0 : 1.0
#       contains(737-700) = True           positi : negati =     11.0 : 1.0
#      contains(generous) = True           positi : negati =     11.0 : 1.0
#        contains(refund) = True           negati : positi =     10.9 : 1.0
#          contains(b737) = True           positi : negati =     10.5 : 1.0
#        contains(superb) = True           positi : negati =     10.3 : 1.0
#          contains(beef) = True           positi : negati =     10.3 : 1.0
#   contains(outstanding) = True           positi : negati =     10.2 : 1.0
#       contains(nervous) = True           positi : negati =      9.7 : 1.0
#     contains(delicious) = True           positi : negati =      9.6 : 1.0
#     contains(attentive) = True           positi : negati =      9.5 : 1.0
#       contains(finally) = True           negati : positi =      9.5 : 1.0
#      contains(equipped) = True           positi : negati =      9.4 : 1.0
#      contains(spotless) = True           positi : negati =      9.4 : 1.0
#          contains(told) = True           negati : positi =      9.2 : 1.0
#          accuracy: 0.828498293515