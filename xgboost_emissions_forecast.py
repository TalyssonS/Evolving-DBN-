from datetime import datetime, timedelta
#import jupyter_contrib_nbextensions
import random
import warnings
import sys 
import logging
import pandas as pd
import time
import numpy as np
import pickle
from tqdm import tqdm
import psycopg2 as pg
import sqlalchemy as sq
import networkx as nx
logging.disable()
if not sys.warnoptions:
    warnings.simplefilter("ignore")
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingClassifier

def errorf (real,forecast):
    error=[]
    for i in range(len(real)):
        error.append(real[i]-forecast[i])
    return error

def open_connection():
    '''
    FUNCTION TO CONNECT TO THE POSTGRESQL DATABASE
    '''
    conn = pg.connect(dbname='postgres', user = 'postgres', password = 123, host = 'localhost')
    return conn
def get_connection():
    '''
    FUNCTION TO CONNECT TO THE POSTGRESQL DATABASE AND RETURN THE SQLACHEMY ENGINE OBJECT
    -----------
    output: object
        SQLACHEMY ENGINE OBJECT - POSTGRESQL DATABASE CONNECTION
    '''
    user = 'postgres'
    password = 123
    host = 'localhost'
    port = 5432
    database = 'postgres'
    return sq.create_engine(url="postgresql://{0}:{1}@{2}:{3}/{4}".format(user, password, host, port, database))


def get_all_dates(pais,horizonte):
    q = '''select distinct cast("Date" as DATE) as datas from pre_processed_data.dbn_features_selected_{pais}{horizonte} order by datas'''.format(pais=pais,horizonte=horizonte)
    conn = open_connection()
    date = pd.read_sql(q,conn)
    conn.close()
    datas = date['datas'].tolist()
    return datas

def get_dataset(pais,date_ini, date_fin,horizonte):
    q = '''select * 
    from pre_processed_data.dbn_features_selected_{pais}{horizonte} where "Date" between '{date_ini}' and '{date_fin}' '''.format(pais=pais,date_ini=date_ini,date_fin=date_fin,horizonte=horizonte)
    conn = open_connection()
    dataset = pd.read_sql(q,conn)
    conn.close()
    return dataset

def get_dataset_allfeatures(pais,date_ini, date_fin, horizonte):
    q = '''select * 
    from pre_processed_data.dbn_{pais}{horizonte} where "Date" between '{date_ini}' and '{date_fin}' '''.format(pais=pais,date_ini=date_ini,date_fin=date_fin,horizonte=horizonte)
    conn = open_connection()
    dataset = pd.read_sql(q,conn)
    conn.close()
    return dataset


def bins_values(pais):
    q = '''select "Emission" from pre_processed_data.bins_{pais} where "Emission" is not null'''.format(pais = pais)
    conn = open_connection()
    df = pd.read_sql(q,conn)
    conn.close()
    return df

def real_values(pais, data):
    q = '''select "Emission" from pre_processed_data.{pais} where "Date" = '{dataf}' '''.format(pais = pais, dataf = (data+timedelta(days = 1)).strftime("%Y/%m/%d"))
    conn = open_connection()
    df = pd.read_sql(q,conn)
    conn.close()
    return df
paises = ['germany','belgium','spain', 'portugal']
horizonte = 3
for pais in paises:
    #initialize auxiliary variables
    k=1 #total days used
    target_variable = 'Emission'
    #xgboost
    timeinferencexgboost = []
    forecast_valuesxgboost = pd.DataFrame()

    #read all available dates
    dates = get_all_dates(pais, horizonte)

    for i in tqdm(dates):
        if i >= dates[6] and i+timedelta(days = 1) in dates:
            #bins
            bins = bins_values(pais)

            #fit dataset (last 7 days)
            fit_data = get_dataset(pais,i-timedelta(days = 6), i, horizonte)
            fit_dataall = get_dataset_allfeatures(pais,i-timedelta(days = 6), i, horizonte)

            #predict data of the entire day
            predict_data_day = get_dataset(pais,i+timedelta(days = 1), i+timedelta(days = 1), horizonte)
            predict_dataall = get_dataset_allfeatures(pais,i+timedelta(days = 1), i+timedelta(days = 1), horizonte)

            #aux
            aux_forexgboost = []
            #dataset to save de forecast values
            forecast_auxxgboost = pd.DataFrame()
            forecast_date = []
            forecast_hour = []

            #predict each point of day i+1
            ti_inf = time.time()
            for h in range(len(predict_data_day)):
                forecast_date.append(i+timedelta(days = 1))
                forecast_hour.append(h)
                predict_data = predict_data_day.iloc[[h]]
                predictall = predict_dataall.iloc[[h]]
                fit_datah = fit_data.loc[0:len(fit_data)-horizonte+h] #tau = horizonte (forecast horizon)

                #drop all variable in time window T+1 (unknown values - future states)
                predict_data.drop(['Date', 'Hour'], axis = 1, inplace = True)
                for c in predict_data.columns:
                    if '-1' not in c:
                        predict_data[c] = predictall[c+str('-1')]
                del predict_data[target_variable]

                #xgboost model
                clf = GradientBoostingClassifier()
                X = fit_datah.copy()
                y = fit_datah[[target_variable]]
                X.drop([target_variable,'Date', 'Hour'], axis = 1, inplace = True)
                clf.fit(X, y)
                xgboost_predict = clf.predict(predict_data)
                for v in xgboost_predict:
                    aux_forexgboost.append((bins[target_variable][v]+bins[target_variable][v+1])/2)
                fit_data = fit_data.append(predict_data_day.loc[h-horizonte:h-horizonte]).reset_index(drop = True) 

            forecast_auxxgboost['Date'] = forecast_date
            forecast_auxxgboost['Hour'] = forecast_hour
            forecast_auxxgboost['Emissions Forecast'] = aux_forexgboost
            real_value = real_values(pais, i)
            forecast_auxxgboost[target_variable] = real_value[target_variable]
            forecast_valuesxgboost = forecast_valuesxgboost.append(forecast_auxxgboost)
            tf_inf = time.time()
            timeinferencexgboost.append(tf_inf-ti_inf)
        #save the results on postgres
        df_time_inferencexgboost = pd.DataFrame()
        df_time_inferencexgboost['tempo'] = timeinferencexgboost
        df_time_inferencexgboost.to_sql(name='time_inference_'+str(pais)+str('xgboost')+str(horizonte), con = get_connection(),schema = 'results', if_exists = 'replace', chunksize = None, index = False)
        forecast_valuesxgboost.to_sql(name='forecast_'+str(pais)+str('xgboost')+str(horizonte), con = get_connection(),schema = 'results', if_exists = 'replace', chunksize = None, index = False)