#!/usr/bin/env python
from itertools import permutations
import time
import networkx as nx
import math

from pgmpy.estimators import StructureEstimator, K2Score
from pgmpy.models import BayesianModel

import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
import time
import math

from pgmpy.estimators import BDeuScore
from pgmpy.estimators import ExhaustiveSearch

### Condition Prob Density Estimation
from pgmpy.inference import VariableElimination
from pgmpy.estimators import BayesianEstimator


class HillClimbSearch(StructureEstimator):
    def __init__(self, data, scoring_method=None, **kwargs):
    
        if scoring_method is not None:
            self.scoring_method = scoring_method
        else:
            self.scoring_method = K2Score(data, **kwargs)

        super(HillClimbSearch, self).__init__(data, **kwargs)

    def _legal_operations(self, model, tabu_list=[], max_indegree=None):
        local_score = self.scoring_method.local_score
        nodes = self.state_names.keys()
        potential_new_edges = (set(permutations(nodes, 2)) -
                               set(model.edges()) -
                               set([(Y, X) for (X, Y) in model.edges()]))

        for (X, Y) in potential_new_edges:  # (1) add single edge
            if nx.is_directed_acyclic_graph(nx.DiGraph(model.edges() + [(X, Y)])):
                operation = ('+', (X, Y))
                if operation not in tabu_list:
                    old_parents = model.get_parents(Y)
                    new_parents = old_parents + [X]
                    if max_indegree is None or len(new_parents) <= max_indegree:
                        score_delta = local_score(Y, new_parents) - local_score(Y, old_parents)
                        yield(operation, score_delta)

        for (X, Y) in model.edges():  # (2) remove single edge
            operation = ('-', (X, Y))
            if operation not in tabu_list:
                old_parents = model.get_parents(Y)
                new_parents = old_parents[:]
                new_parents.remove(X)
                score_delta = local_score(Y, new_parents) - local_score(Y, old_parents)
                
                yield(operation, score_delta)

        for (X, Y) in model.edges():  # (3) flip single edge
            new_edges = model.edges() + [(Y, X)]
            new_edges.remove((X, Y))
            if nx.is_directed_acyclic_graph(nx.DiGraph(new_edges)):
                operation = ('flip', (X, Y))
                if operation not in tabu_list and ('flip', (Y, X)) not in tabu_list:
                    old_X_parents = model.get_parents(X)
                    old_Y_parents = model.get_parents(Y)
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = old_Y_parents[:]
                    new_Y_parents.remove(X)
                    if max_indegree is None or len(new_X_parents) <= max_indegree:
                        score_delta = (local_score(X, new_X_parents) +
                                       local_score(Y, new_Y_parents) -
                                       local_score(X, old_X_parents) -
                                       local_score(Y, old_Y_parents))
                        yield(operation, score_delta)
    def estimate(self, start=None, extra_tabu_list=[] ,tabu_length=0, max_indegree=None):
        epsilon = 1e-8
        itermax=100
        iterr=0
        nodes = self.state_names.keys()
        if start is None:
            start = BayesianModel()
            start.add_nodes_from(nodes)
        #elif not isinstance(start, BayesianModel) or not set(start.nodes()) == set(nodes):
         #   raise ValueError("'start' should be a BayesianModel with the same variables as the data set, or 'None'.")
        else:
            start.add_nodes_from(nodes)
        tabu_list = []
        current_model = start
        stop_time = 1800
        inicial_time = time.time()
        while True:
            best_score_delta = 0
            best_operation = None
            scorit=[]
            
            #print(iterr)
            #print(current_model.edges())
            for operation, score_delta in self._legal_operations(current_model, tabu_list+extra_tabu_list, max_indegree):
                
                if score_delta > best_score_delta:
                    best_operation = operation
                    best_score_delta = score_delta
                scorit.append(best_score_delta)
            current_time = time.time()
            if best_operation is None or best_score_delta < epsilon or iterr > itermax or current_time-inicial_time>=stop_time:
                break
            elif best_operation[0] == '+':
                current_model.add_edge(*best_operation[1])
                tabu_list = ([('-', best_operation[1])] + tabu_list)[:tabu_length]
            elif best_operation[0] == '-':
                current_model.remove_edge(*best_operation[1])
                tabu_list = ([('+', best_operation[1])] + tabu_list)[:tabu_length]
            elif best_operation[0] == 'flip':
                X, Y = best_operation[1]
                current_model.remove_edge(X, Y)
                current_model.add_edge(Y, X)
                tabu_list = ([best_operation] + tabu_list)[:tabu_length]
            iterr=iterr+1
        return current_model,scorit
    
class TimedFun(HillClimbSearch):
    """
    Classe custom de otimização
    Restringida por tempo de execução
    """
    def __init__(self, fun, stop_after=50):
        super().__init__()
        
        self.fun_in = fun
        self.started = False
        self.stop_after = stop_after
        self.last = 0

    def fun(self, x):
        if self.started is False:
            self.started = time.time()
        elif abs(time.time() - self.started) >= self.stop_after:
            raise ValueError("Time is over.")
        
        if self.started:
            t = int(abs(time.time() - self.started))
            if (t >= 600) and (int(t/600) != self.last):
                self.last = int(t/600)
                self.opt_logger.info('Tempo execução optimization: [%.1f] minutos' % (int(t/600)*10))
            
        self.fun_value = self.fun_in(x)
        self.x = x
        return self.fun_value

def EdgesModel(data, FullOpt):
    ## score definition
    bdeu = BDeuScore(data, equivalent_sample_size=10)
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

    #return best_model.edges(),val[1]
    return best_model.edges()

def ModelInference(best_model, data):
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
