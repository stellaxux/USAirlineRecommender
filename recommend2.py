# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 20:26:18 2016

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

# for each airline, reconstruct the dataframe so that items are service categories i.e. sear_comfort_rating, cabin_staff_rating, food_beverages_rating, etc
# train_aa = airline_us[airline_us['airline_name'] == 'american-airlines']
train = pd.melt(airline_us, id_vars=['author','airline_name'], value_vars=['seat_comfort_rating','cabin_staff_rating', 'food_beverages_rating','inflight_entertainment_rating'], var_name='category', value_name='rating')
train = train.dropna()
train['airline_category'] = train.airline_name.str.cat(train.category, sep ='_')
train = train.drop(['airline_name', 'category'], axis=1)
train_data = graphlab.SFrame(train)

# A Collaborative Filtering Model
# Train Model
item_sim_model = graphlab.item_similarity_recommender.create(train_data, user_id='author', item_id='airline_category', target='rating', similarity_type='pearson')
#Make Recommendations:
item_sim_recomm = item_sim_model.recommend(k=10)
item_sim_recomm.print_rows(num_rows=50)