# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 15:52:41 2016

Thinkful capstone project

@author: Xin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

## read in data by month
#data = pd.read_csv('/Users/Xin/project/thinkful/capstone/On_Time_On_Time_Performance_2016/On_Time_On_Time_Performance_2016_1.csv', low_memory=False)
#    
#data['FlightDate'] = pd.to_datetime(data['FlightDate'])
#
#col = [ 'FlightDate', 'UniqueCarrier', 'Origin', 'OriginCityName', 'OriginState', 'Dest', \
#        'DestCityName', 'DestState', 'CRSDepTime', 'DepTime', 'DepDelayMinutes', \
#        'DepDel15', 'CRSArrTime', 'ArrTime', 'ArrDelayMinutes', 'ArrDel15', \
#        'Cancelled', 'CancellationCode', 'Distance', 'CarrierDelay', \
#        'WeatherDelay', 'NASDelay', 'SecurityDelay', 'LateAircraftDelay']
#        
#month_data = data[col]
#
## consolidate monthly data to a single csv file
#month_data.to_csv('ontime_performance2016.csv', mode='a', header=False, index=False)

data2015 = pd.read_csv('ontime_performance2015.csv', low_memory=False)

