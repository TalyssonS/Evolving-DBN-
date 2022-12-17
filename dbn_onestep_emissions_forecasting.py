from pgmpy.estimators import HillClimbSearch
from datetime import datetime, timedelta
from pgmpy.inference import VariableElimination
import random
import warnings
import sys 
import pandas as pd
import time
import numpy as np
import pickle
from tqdm import tqdm
import psycopg2 as pg
import sqlalchemy as sq
import networkx as nx
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import BDeuScore 
from pgmpy.estimators import K2Score
from pgmpy.estimators import AICScore
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
import matplotlib.pyplot as plt
import logging
logging.disable()
if not sys.warnoptions:
    warnings.simplefilter("ignore")

def smooth(y, box_pts):
    vi = y[0]
    vf = y[len(y)-1]
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    y_smooth[0] = vi
    y_smooth[len(y_smooth)-1] = vf
    return y_smooth

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

def blanket(model,variable):
    '''
    Function to extract the Markov Blanket of the target variable (reduces the structure)
    -----------
    input:
    model: list
        list of edges
        
    variable: str
        name of the target variable
        
    output:
    blanket: list
        list of edges 
    '''
    blanket=[]
    sons=[]
    for i in model:
        if i[0]==variable or i[1]==variable:
            blanket.append(i)
        if i[0]==variable:
            sons.append(i[1])
    for i in model:
        if i[1] in sons and i[0]!=variable:
            blanket.append(i)
    return blanket

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

def main(pais,horizonte):
    #initialize auxiliary variables
    k=1 #total days used
    target_variable = 'Emission'
    timemodel = []
    timeinference = []
    forecast_values = pd.DataFrame()

    #read all available dates
    dates = get_all_dates(pais,horizonte)
    
    #dataset to learn the model
    data_learn = get_dataset(pais,dates[0], dates[0]+timedelta(days = 6),horizonte)
    #structural learning with the dataset of day i
    data_learn.drop(['Date','Hour'], axis = 1, inplace = True)        
    est = HillClimbSearch(data_learn)
    best_model = est.estimate(scoring_method=K2Score(data_learn),  show_progress=False)
    best_model = list(best_model.edges())

    #get the markov blanket
    edges = blanket(best_model, target_variable)
    
    print(edges)

    #begin the forecast experiment
    for i in tqdm(dates):
        #dataset to save de forecast values
        forecast_aux = pd.DataFrame()
        forecast_date = []
        forecast_hour = []
        forecast_v = []

        #forecast initial in day 7 (fit from 00 until 06)
        
        inicio = dates[6] #begin the forecast on day 7 (6+1)
        if i >= inicio and i+timedelta(days = 1) in dates:
            #bins
            bins = bins_values(pais)
            
            #fit dataset (last 7 days)
            fit_data = get_dataset(pais,i-timedelta(days = 6), i,horizonte)
            fit_dataall = get_dataset_allfeatures(pais,i-timedelta(days = 6), i,horizonte)

            #predict data of the entire day
            predict_data_day = get_dataset(pais,i+timedelta(days = 1), i+timedelta(days = 1),horizonte)
            predict_dataall = get_dataset_allfeatures(pais,i+timedelta(days = 1), i+timedelta(days = 1),horizonte)

            #detects independent variables
            independentes=[]
            for col in fit_data.columns:
                if col not in list(map(lambda x: x[0],edges))+list(map(lambda x: x[1],edges)):
                    independentes.append(col)

            #drop independent columns
            fit_data.drop(independentes, axis=1, inplace = True)
            predict_data_day.drop(independentes, axis=1, inplace = True)

            #transform data in levels (limitation of PGMPY)
            levels = {}
            aux = fit_dataall.copy()
            aux = aux.append(predict_dataall)
            for var in aux.columns:
                levels[var] = set(aux[var])
                fit_dataall[var] = fit_dataall[var].replace(levels[var], np.arange(0,len(levels[var])))
                predict_dataall[var] = predict_dataall[var].replace(levels[var], np.arange(0,len(levels[var])))
                if var in fit_data.columns:
                    fit_data[var] = fit_data[var].replace(levels[var], np.arange(0,len(levels[var])))
                    predict_data_day[var] = predict_data_day[var].replace(levels[var], np.arange(0,len(levels[var])))
            predict_data_day = predict_data_day.astype(int)
            
            #Using the edges, get the bayesian model object
            model = BayesianNetwork(edges)

            #aux
            aux_fore = []
        
            #predict each point of day i+1
            ti_inf = time.time()
            for h in range(len(predict_data_day)):
                forecast_date.append(i+timedelta(days = 1))
                forecast_hour.append(h)
                predict_data = predict_data_day.iloc[[h]]
                predictall = predict_dataall.iloc[[h]]
                fit_datah = fit_data.loc[0:len(fit_data)-horizonte+h] #tau = horizonte (forecast horizon)
                                
                #fit the bayesian model to get de CPTs
                model.fit(fit_datah,n_jobs = 1)
                #model.get_cpds(node = target_variable)

                #drop all variable in time window T+1 (unknown values - future states)
                for c in predict_data.columns:
                    if '-1' not in c:
                        predict_data[c] = predictall[c+str('-1')]
                del predict_data[target_variable]
                
                #solve limitation of unknown level 
                for col in predict_data.columns:
                    predict_data[col][predict_data[col]>=len(set(fit_datah[col]))] = len(set(fit_datah[col]))-1
                y_pred = model.predict(predict_data,n_jobs = 1)
                y_pred[target_variable] = y_pred[target_variable].replace(np.arange(0,len(levels[target_variable])),levels[target_variable])
                for v in y_pred[target_variable]:
                    aux_fore.append((bins[target_variable][v]+bins[target_variable][v+1])/2)
                fit_data = fit_data.append(predict_data_day.loc[h-horizonte:h-horizonte]).reset_index(drop = True) 
            forecast_aux['Date'] = forecast_date
            forecast_aux['Hour'] = forecast_hour
            forecast_aux['Emissions Forecast'] = smooth(aux_fore,3)
            real_value = real_values(pais, i)
            forecast_aux[target_variable] = real_value[target_variable]
            forecast_values = forecast_values.append(forecast_aux)
            tf_inf = time.time()
            timeinference.append(tf_inf-ti_inf)
        k = k+1
    #save the results on postgres
    df_time_inference = pd.DataFrame()
    df_time_inference['tempo'] = timeinference

    df_time_inference.to_sql(name='time_inference_final_'+str(pais)+str('dbn_onestep')+str(horizonte), con = get_connection(),schema = 'results', if_exists = 'replace', chunksize = None, index = False)
    forecast_values.to_sql(name='forecast_final_'+str(pais)+str('dbn_onestep')+str(horizonte), con = get_connection(),schema = 'results', if_exists = 'replace', chunksize = None, index = False)
paises = ['alemanha','belgica','espanha', 'portugal']
for pais in paises:
    main(pais,3)
