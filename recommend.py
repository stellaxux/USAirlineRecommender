# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 23:03:47 2016

@author: Xin
"""

import pandas as pd
import graphlab

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

# select author, airline_name, ratings for SFrame
airline_us = airline_us[['author', 'airline_name', 'overall_rating', 'seat_comfort_rating','cabin_staff_rating', 'food_beverages_rating','inflight_entertainment_rating']]
# drop rows with missing data
rating_all = airline_us.dropna(subset=['overall_rating'])
rating_seat = airline_us.dropna(subset=['seat_comfort_rating'])
rating_staff = airline_us.dropna(subset=['cabin_staff_rating'])
rating_food = airline_us.dropna(subset=['food_beverages_rating'])
rating_ent = airline_us.dropna(subset=['inflight_entertainment_rating'])

train = rating_all[:4000]
test = rating_all[4000:]

# convert tables into SFrames using graphlab
train_data = graphlab.SFrame(train)
test_data = graphlab.SFrame(test)

# A Simple Popularity Model
popularity_model = graphlab.popularity_recommender.create(train_data, user_id='author', item_id='airline_name', target='overall_rating')

#Get recommendations for first 5 users and print them
#k=5 specifies top 5 recommendations to be given
popularity_recomm = popularity_model.recommend(k=5)
popularity_recomm.print_rows(num_rows=25)

# verified by checking the airlines with highest mean rating
train.groupby(by='airline_name')['overall_rating'].mean().sort_values(ascending=False)

# A Collaborative Filtering Model
#Train Model
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='author', item_id='airline_name', target='overall_rating', similarity_type='pearson')

#Make Recommendations:
item_sim_recomm = item_sim_model.recommend(k=5)
item_sim_recomm.print_rows(num_rows=25)

model_performance = graphlab.compare(test_data, [popularity_model, item_sim_model])
graphlab.show_comparison(model_performance,[popularity_model, item_sim_model])
