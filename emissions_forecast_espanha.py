import DefModules as DM
from datetime import datetime, timedelta
FullOpt =  True
from pgmpy.inference import VariableElimination
import jupyter_contrib_nbextensions
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
from pgmpy.models import BayesianModel
from pgmpy.estimators import HillClimbSearch
FullOpt =  True #True é para usar HC e false busca exaustiva
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator
from pgmpy.estimators import BdeuScore
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler

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

def verifica_remove_ciclos(edges):
    '''
    Function to verify if the edges is a DAG and to try remove cycles
    -----------
    input:
    edges: list
        list of edges
                
    output:
    blanket: list
        list of edges 
    '''
    edgesdag = edges #recebe o próprio modelo
    #Verifica se tem ciclos e tenta remover invertendo uma aresta
    if ~nx.is_directed_acyclic_graph(nx.DiGraph(edgesdag[:])):
        for i in edgesdag:  # (3) flip single edge
            edges2 = edgesdag.copy()
            edges2.extend([i[::-1]])
            new_edges = edges2.copy()
            new_edges.remove(i)
            if nx.is_directed_acyclic_graph(nx.DiGraph(new_edges[:])):
                edgesdag = new_edges.copy()
                break
    #Verifica se tem ciclos e tenta remover invertendo duas arestas
    if ~nx.is_directed_acyclic_graph(nx.DiGraph(edgesdag[:])):
        for i in edgesdag:
            for j in edgesdag:# (3) flip two edges
                if i != j:
                    edges2 = edgesdag.copy()
                    edges2.extend([i[::-1]])
                    edges2.extend([j[::-1]])
                    new_edges = edges2.copy()
                    new_edges.remove(i)
                    new_edges.remove(j)
                    if nx.is_directed_acyclic_graph(nx.DiGraph(new_edges[:])):
                        edgesdag = new_edges.copy()
                        breaker = True
                        break
            if breaker:
                break
    #Verifica se tem ciclos e tenta remover invertendo uma aresta e excluindo uma aresta
    if ~nx.is_directed_acyclic_graph(nx.DiGraph(edgesdag[:])):
        for i in edgesdag:
            for j in edgesdag:# (3) flip two edges
                if i != j:
                    edges2 = edgesdag.copy()
                    edges2.extend([i[::-1]])
                    new_edges = edges2.copy()
                    new_edges.remove(i)
                    new_edges.remove(j)
                    if nx.is_directed_acyclic_graph(nx.DiGraph(new_edges[:])):
                        edgesdag = new_edges.copy()
                        breaker = True
                        break
            if breaker:
                break
    return edgesdag
def get_all_dates(pais):
    q = '''select distinct cast("Date" as DATE) as datas from pre_processed_data.dbn_features_selected_{pais} order by datas'''.format(pais=pais)
    conn = open_connection()
    date = pd.read_sql(q,conn)
    conn.close()
    datas = date['datas'].tolist()
    return datas

def get_dataset(pais,date_ini, date_fin):
    q = '''select * 
    from pre_processed_data.dbn_features_selected_{pais} where "Date" between '{date_ini}' and '{date_fin}' '''.format(pais=pais,date_ini=date_ini,date_fin=date_fin)
    conn = open_connection()
    dataset = pd.read_sql(q,conn)
    conn.close()
    return dataset

def get_dataset_allfeatures(pais,date_ini, date_fin):
    q = '''select * 
    from pre_processed_data.dbn_{pais} where "Date" between '{date_ini}' and '{date_fin}' '''.format(pais=pais,date_ini=date_ini,date_fin=date_fin)
    conn = open_connection()
    dataset = pd.read_sql(q,conn)
    conn.close()
    return dataset

def update_edges_frequencies(best_model, edges_possibilities, edges_frequency):
    if not edges_possibilities:
        edges_possibilities = best_model
        for p in range(len(edges_possibilities)):
            edges_frequency.append(1)
    else:
        for v in range(len(best_model)):
            if best_model[v] not in edges_possibilities:
                edges_possibilities.append(best_model[v])
                edges_frequency.append(1)
            else:
                for f in range(len(edges_possibilities)):
                    if best_model[v] == edges_possibilities[f]:
                        edges_frequency[f]=edges_frequency[f]+1
    return edges_possibilities, edges_frequency

def update_threshold_select_edges(k, edges_possibilities, edges_frequency):
    fth = 1/3+np.sqrt(2/k)
    if fth>0.4:
        fth=0.4
    edges_frequency_v=[edges_frequency[i]/k for i in range(len(edges_frequency))]
    edges=[]
    for i in range(len(edges_possibilities)):
        if edges_frequency_v[i]>=fth and edges_possibilities[i] not in edges:
            if edges_possibilities[i][::-1] not in edges_possibilities:
                edges.append(edges_possibilities[i])
            else: 
                if edges_frequency_v[i] > edges_frequency_v[edges_possibilities.index(edges_possibilities[i][::-1])]:
                    edges.append(edges_possibilities[i])
                else: 
                    edges.append(edges_possibilities[i][::-1])
        elif edges_frequency_v[i]<fth:
            for j in range(len(edges_possibilities)):
                if edges_possibilities[i]==edges_possibilities[j][::-1]:
                    if edges_frequency_v[i]+edges_frequency_v[j]>=fth:
                        if edges_frequency_v[i]>edges_frequency_v[j]:
                            edges.append(edges_possibilities[i])
                        if edges_frequency_v[i]<edges_frequency_v[j]:
                            edges.append(edges_possibilities[j])
                        if edges_frequency_v[i]==edges_frequency_v[j]:
                            auxci=0
                            auxcj=0
                            for s in range(len(edges)):
                                if edges[s]==edges_possibilities[i]:
                                    auxci=auxci+1
                                if edges[s]==edges_possibilities[j]:
                                    auxcj=auxcj+1
                            if auxci>0:
                                edges.append(edges_possibilities[i])
                            elif auxcj>0:
                                edges.append(edges_possibilities[j])
                            else: 
                                import random
                                edges.append(random.choice([edges_possibilities[i],edges_possibilities[j]]))
    edges = list(set(edges))
    return edges

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

def main(pais):
    #initialize auxiliary variables
    k=1 #total days used
    target_variable = 'Emission'
    edges_possibilities = []
    edges_frequency = []
    timemodel = []
    timeinference = []
    forecast_values = pd.DataFrame()

    #read all available dates
    dates = get_all_dates(pais)

    #begin the forecast experiment
    for i in tqdm(dates):
        #dataset to save de forecast values
        forecast_aux = pd.DataFrame()
        forecast_date = []
        forecast_hour = []
        forecast_v = []
        #dataset to learn the model
        data_learn = get_dataset(pais,i, i+timedelta(days = 0))
        #structural learning with the dataset of day i
        data_learn.drop(['Date','Hour'], axis = 1, inplace = True)
        ti = time.time()
        best_model = DM.EdgesModel(data_learn, FullOpt)[0]

        #get the markov blanket
        best_model = blanket(best_model, target_variable)

        #update the edges frequencies
        edges_possibilities, edges_frequency = update_edges_frequencies(best_model, edges_possibilities, edges_frequency)

        #update threshold and select the edges
        edges = update_threshold_select_edges(k, edges_possibilities, edges_frequency)

        tf = time.time()
        timemodel.append(tf-ti)
        #forecast initial in day 8 (fit from 01 until 07)
        if i >= dates[6]:
            #bins
            bins = bins_values(pais)
            
            #fit dataset (last 7 days)
            fit_data = get_dataset(pais,i-timedelta(days = 6), i)
            fit_dataall = get_dataset_allfeatures(pais,i-timedelta(days = 6), i)

            #predict data of the entire day
            predict_data_day = get_dataset(pais,i+timedelta(days = 1), i+timedelta(days = 1))
            predict_dataall = get_dataset_allfeatures(pais,i+timedelta(days = 1), i+timedelta(days = 1))

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
            model = BayesianModel(edges)

            #aux
            aux_fore = []
        
            #predict each point of day i+1
            ti_inf = time.time()
            for h in range(len(predict_data_day)):
                forecast_date.append(i+timedelta(days = 1))
                forecast_hour.append(h)
                predict_data = predict_data_day.iloc[[h]]
                predictall = predict_dataall.iloc[[h]]
                fit_datah = fit_data.loc[0:len(fit_data)-3+h] #tau = 3 (forecast horizon)
                
                #fit the bayesian model to get de CPTs
                model.fit(fit_datah)
                model.get_cpds(node = target_variable)

                #drop all variable in time window T+1 (unknown values - future states)
                for c in predict_data.columns:
                    if '-1' not in c:
                        predict_data[c] = predictall[c+str('-1')]
                del predict_data[target_variable]
                
                #solve limitation of unknown level 
                for col in predict_data.columns:
                    predict_data[col][predict_data[col]>=len(set(fit_datah[col]))] = len(set(fit_data[col]))-1
                y_pred = model.predict(predict_data)
                y_pred[target_variable] = y_pred[target_variable].replace(np.arange(0,len(levels[target_variable])),levels[target_variable])
                for v in y_pred[target_variable]:
                    aux_fore.append((bins[target_variable][v]+bins[target_variable][v+1])/2)
                fit_data = fit_data.append(predict_data_day.loc[h-3:h-3]).reset_index(drop = True) 
            forecast_aux['Date'] = forecast_date
            forecast_aux['Hour'] = forecast_hour
            forecast_aux['Emissions Forecast'] = smooth(aux_fore,5)
            real_value = real_values(pais, i)
            forecast_aux[target_variable] = real_value[target_variable]
            forecast_values = forecast_values.append(forecast_aux)
            tf_inf = time.time()
            timeinference.append(tf_inf-ti_inf)
        k = k+1
    #save the results on postgres
    df_edges = pd.DataFrame()
    df_edges['edges'] = edges_possibilities
    df_edges['frequencia'] = edges_frequency
    df_edges['total days'] = k-1
    df_time_model = pd.DataFrame()
    df_time_model['tempo'] = timemodel
    df_time_inference = pd.DataFrame()
    df_time_inference['tempo'] = timeinference
    
    df_edges.to_sql(name='edges_frequency_'+str(pais), con = get_connection(),schema = 'results', if_exists = 'replace', chunksize = None, index = False)
    df_time_model.to_sql(name='time_model_'+str(pais), con = get_connection(),schema = 'results', if_exists = 'replace', chunksize = None, index = False)
    df_time_inference.to_sql(name='time_inference_'+str(pais), con = get_connection(),schema = 'results', if_exists = 'replace', chunksize = None, index = False)
    forecast_values.to_sql(name='forecast_'+str(pais), con = get_connection(),schema = 'results', if_exists = 'replace', chunksize = None, index = False)
main('espanha')