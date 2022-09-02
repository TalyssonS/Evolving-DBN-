import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import time
import math
from sklearn.metrics.cluster import normalized_mutual_info_score

def EdgesModel(data, FullOpt):
    from pgmpy.estimators import BdeuScore
    from pgmpy.models import BayesianModel
    from BDEsolve import HillClimbSearch
    from pgmpy.estimators import ExhaustiveSearch

    ## score definition
    bdeu = BdeuScore(data, equivalent_sample_size=10)
    if FullOpt == True: #HC
        hc = HillClimbSearch(data, scoring_method= bdeu)
        #val = hc.estimate(initial_model,extra_tabu_list)    
        val = hc.estimate() 
        best_model = val[0]
    else: #Exhaustive
        es = ExhaustiveSearch(data, scoring_method= bdeu)
        best_model = es.estimate()
        print("All DAGs by score:")
        f = open('out.txt', 'w')
        for score, dag in reversed(es.all_scores()):
            f.write(str(score) + ' ' + str(dag.edges())+'\n')  # Python 3.x
        with open('model.json', 'w') as outfile:
            json.dump( best_model.edges(), outfile)

    return best_model.edges(),val[1]

def ModelInference(best_model, data):
    ### Condition Prob Density Estimation
    from pgmpy.inference import VariableElimination
    from pgmpy.estimators import BayesianEstimator
    #from pgmpy.estimators import MaximumLikelihoodEstimator
    from pgmpy.models import BayesianModel

    model = BayesianModel(best_model) #.edges())
    model.fit(data)
    model.get_cpds()

    estimator = BayesianEstimator(model, data)
    #estimator = MaximumLikelihoodEstimator(model, data)
    cpds = estimator.get_parameters()

    for cpd in cpds:
        model.add_cpds(cpd)
    model_inference = VariableElimination(model)
    ### end of Condition Prob Density Estimation

    ###  dic captures the variables from dataset
    labels = list(data.columns)
    dic = {}
    for label in labels:
        for i in range(len(data)):
            if label not in dic.keys():
                dic[label] = []
            if data[label][i] not in dic[label]:
                dic[label].append(data[label][i])
    for label in labels:
        dic[label].sort()
    ### end dic
    print(cpd)
    return model_inference, dic
