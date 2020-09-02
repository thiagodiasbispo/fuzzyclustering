from itertools import chain
import pickle

import numpy as np
import pandas as pd

def get_hard_patitions(membership_degree):
    """
        membership_degree: numpy array of shape N x K
    """
    
    members = []
    K = membership_degree.shape[1]
    
    # Obtendo o índice do grupo em que cada elemento possui maior valor de pertencimento
    membership = membership_degree.argsort(axis=1)[:, -1]
    members = [np.where(membership == k)[0] for k in range(K)]

    return members, membership

def get_instances_class():
    y_true = np.empty(2000)
    y_true[0:200] = 0
    y_true[200:400] = 1
    y_true[400:600] = 2
    y_true[600:800] = 3
    y_true[800:1000] = 4
    y_true[1000:1200] = 5
    y_true[1200:1400] = 6
    y_true[1400:1600] = 7
    y_true[1600:1800] = 8
    y_true[1800:2000] = 9
    
    return y_true

def calc_partition_coefficient(membership_degree):
    n = membership_degree.shape[0]
    membership_degree = np.power(membership_degree, 2)
    return membership_degree.sum(axis = 1).sum()/n

def calc_modified_partition_coefficient(membership_degree):
    c = membership_degree.shape[1]
    vpc = calc_partition_coefficient(membership_degree)
    return 1 - (c/(c-1))*(1-vpc)

def calc_partition_entropy(membership_degree):
    n = membership_degree.shape[0]
    membership_degree = np.log10(membership_degree) * membership_degree
    return -membership_degree.sum(axis = 1).sum()/n

def import_best_result(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)

def import_fuzzy_partitions_from_csv(file_name):
    return pd.read_csv(file_name, decimal=',')
