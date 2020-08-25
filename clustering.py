from itertools import chain
import pickle

import numpy as np
import pandas as pd

def get_hard_patitions(membership_degree):
    """
        membership_degree: numpy array of shape N x K
    """
    
    u = membership_degree
    members = []
    K = membership_degree.shape[1]
    
    # Obtendo o índice do grupo em que cada elemento possui maior valor de pertencimento
    element_index_membership = u.argsort(axis=1)[:, -1]
    
    index_class = {}
    for k in range(K):
        # Para cada grupo k, extrai quais dos elementos possui maior grau de pertencimento para ele
        memb = np.where(element_index_membership == k)[0]
        members.append(memb)
        index_class.update({m:k for m in memb})
    
    predicted_classes = [index_class[m] for m in sorted(index_class.keys())]
    
    return members, predicted_classes

def get_instances_class(qtd=10):
    return list(chain(*[[i]*200 for i in range(qtd)]))

def calc_partition_entropy(membership_degree):
    n = membership_degree.shape[0]
    membership_degree = np.log10(membership_degree) * membership_degree
    return -membership_degree.sum(axis = 1).sum()/n

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
