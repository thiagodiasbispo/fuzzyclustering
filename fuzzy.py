import os
import functools
import operator
import math
import random
import sys
import multiprocessing as mp
import pickle
from itertools import product

import pandas as pd
import numpy as np
from sklearn.metrics import adjusted_rand_score, f1_score, accuracy_score

import clustering


def cluster_matching_function(weight_matrix,
                              cluster_number,
                              element,
                              prototypes,
                              dissimilarity_matrices):
    """
        :params: weight_matrix: numpy array-like 
                    matriz K x P de pesos das matrizes de dissimilaridades por cluster
                cluster_number: int
                    Número do cluster em questão
                element: int
                    Índice do elemento (entre 0 e N-1)
                prototypes: list like
                    Lista de tamanho K dos protótipos de cada cluster
                dissimilarity_matrices: lista de numpy array
                    Lista de matrizes de dissimilaridade

        :return: float

    """

    # Criando aliases compatíveis com as variáveis da fórmula
    k = cluster_number
    D = dissimilarity_matrices
    p = len(D)
    Gk = prototypes[k]
    l = weight_matrix

    dissimilarities_sum = np.array([dj[element, Gk].sum() for dj in D])

    return np.dot(l[k], dissimilarities_sum)


def objective_function(clusters_qtd,
                       elements_qtd,
                       adequacy_criterion,
                       m,
                       weight_matrix,
                       prototypes,
                       dissimilarity_matrices):
    """
        :params: clusters_qtd: int
                    Quantidade total de clusters
                elements_qtd: int
                    Quantidade de elementos da base de dados
                adequacy_criterion: numpy array-like
                    Matriz u de tamanho N x K contendo a índice de adequação 
                    de cada elemente a cada cluster
                m: int
                    Fator de ponderação do índice de adequação
                weight_matrix:
                     matriz K x P de pesos das matrizes de dissimilaridades por cluster
                prototypes: list like
                    Lista de tamanho K dos protótipos de cada cluster
                dissimilarity_matrices: lista de numpy array
                    Lista de matrizes de dissimilaridade

        :return: float

    """

    u = np.power(adequacy_criterion, m) # Resolvendo a exponeciação de u de uma vez só
    l = weight_matrix
    D = dissimilarity_matrices
    K = clusters_qtd
    G = prototypes
    N = elements_qtd
    match = cluster_matching_function # Criando um alias para reduzir o nome da função de matching
  
    J = np.array([np.array([u[i, k] * match(l, k, i, G, D) for i in range(N)]).sum() 
          for k in range(K)])


    return J.sum()

def get_prototypes(elements_qtd,
                       q,
                       m,
                       s,
                       cluster_number,
                       adequacy_criterion,
                       dissimilarity_matrices,
                       weight_matrix):
    G = []
    k = cluster_number
    D = dissimilarity_matrices
    u = np.power(adequacy_criterion, m)
    l = np.power(weight_matrix, s)
    N = elements_qtd
    P = len(D)
    
    while (len(G) != q):
        menor_soma = 999999
        menor_indice = None
        
        for h in range(N): 
            if h in G:
                continue
            
            dists_p = np.array([D[j][:, h] * l[k,j] for j in range(P)]) #shape: NxP
            sums_p = dists_p.sum(axis=0)
            soma = np.dot(u[:, k], sums_p)

            #soma = sum([u[i,k] * sum([l[k,j] * D[j][i,h] for j in range(P)]) for i in  range(N)])

            #print(f"Somas: {soma2} {soma}")

            if soma < menor_soma:
                menor_soma = soma
                menor_indice = h
                 
        G.append(menor_indice)
        
    return G
            
def compute_relevance_weights(clusters_qtd,
                              dissimilarity_matrices,
                              prototypes,
                              elements_qtd,
                              adequacy_criterion,
                              m):
    """
        :params:
                clusters_qtd: int
                    Quantidade total de clusters
                dissimilarity_matrices: lista de numpy array
                    Lista de matrizes de dissimilaridade
                prototypes: list like
                    Lista de tamanho K dos protótipos de cada cluster
                elements_qtd: int
                    Quantidade de elementos da base de dados
                adequacy_criterion: numpy array-like
                    Matriz u de tamanho N x K contendo a índice de adequação 
                    de cada elemente a cada cluster
                m: int
                    Fator de ponderação do índice de adequação

        :return: numpy array of shape K x P

    """

    D = dissimilarity_matrices
    P = len(D)
    G = prototypes
    K = clusters_qtd
    N = elements_qtd
    u = np.power(adequacy_criterion, m)
    l = np.zeros((K, P))

    def match(element, Dh, Gk):
        """
            Função auxiliar para cálculo de match entre um elemento 
            qualquer, os protótipos G de um cluster específico e uma matriz 
            de similaridade específica Dh.
        """

        return Dh[element, Gk].sum()

    for k in range(K):
        # Calculado o somatório do numerador da equação à esquerda da igualdade
        weight_diss_sum1 = np.array([np.array([u[i, k] * match(i, D[h], G[k]) for i in range(N)]).sum()
                            for h in range(P)])

        weight_diss_sum_prod = np.power(weight_diss_sum1.prod(), 1/P)

        for j in range(P):
     
            # Calculado o somatório do denominador da equação à esquerda da igualdade
            weight_diss_sum2 = np.array([u[i, k] * match(i, D[j], G[k])
                                    for i in range(N)]).sum()
            

            # Executando a divisão da fração à esquerda da equação
            l[k, j] = weight_diss_sum_prod / weight_diss_sum2

    return l


# %load -s compute_membership_degree 'fuzzy.py'
def compute_membership_degree(weight_matrix,
                              prototypes,
                              clusters_qtd,
                              dissimilarity_matrices,
                              elements_qtd,
                              m):
    """
        :params: weight_matrix: numpy array-like 
                    matriz K x P de pesos das matrizes de dissimilaridades por cluster
                prototypes: list like
                    Lista de tamanho K dos protótipos de cada cluster
                clusters_qtd: int
                    Quantidade total de clusters
                dissimilarity_matrices: lista de numpy array
                    Lista de matrizes de dissimilaridade
                elements_qtd: int
                    Quantidade de elementos da base de dados
                m: int
                    Fator de ponderação do índice de adequação

        :return: numpy array NxK

    """
        

    K = clusters_qtd
    G = prototypes
    D = dissimilarity_matrices
    l = weight_matrix
    P = len(D)
    N = elements_qtd
    u = np.zeros((N, K))
    
    match = cluster_matching_function # Criando um alias para reduzir o nome da função de matching

    def ratio(element, k, h):
        r = match(l, k, element, G, D) / match(l, h, element, G, D)
        #r1 = np.array([l[k,j] * (D[j][element, G[k]].sum()) for j in range(P)]).sum()
        #r2 = np.array([l[h,j] * (D[j][element, G[h]].sum()) for j in range(P)]).sum()
        #r3 = r1/r2
        #print(f"Ratio: {r} {r3}")
        return np.power(r, 1/(m-1))

    for i in range(N):
        for k in range(K):
            outter_sum = np.array([ratio(i, k, h) for h in range(K)]).sum()
            u[i, k] = 1/outter_sum

    return u

def random_prototypes(K, N, q, seed):
    elements = set(range(N))
    protos = []
    random.seed(seed)
    
    for k in range(K):
        protos.append(random.sample(elements, q))
        elements -= set(protos[-1])

    return protos

def assert_relevance_weights_prod_one(l):
    # l.shape == (K,P)
    prods_p = l.prod(axis=1)
    assert round(prods_p.sum()) == l.shape[0], f"O produto dos pesos de relevância não é igual a 1 ({prods_p.sum()})"

def assert_membership_degree_sum_one(u):
     # u.shape == (N,K)
    sums_k = u.sum(axis=1)
    assert round(sums_k.sum()) == u.shape[0], f"A soma dos pesos de relevância não é igual a 1 ({sums_k.sum()})"

def random_prototypes2(K, N, q, seed):
    random.seed(seed)
    protos = np.array(random.sample(range(N), K*q)).reshape(K, q)
    return list(protos)


def executar_treinamento(dissimilarity_matrices,
                       elements_qtd,
                       K=10,
                       m=1.6,
                       T=150,
                       epsilon=10e-10,
                       q=2, 
                       seed=13082020,
                       prototipos = None):

    D = dissimilarity_matrices
    N = elements_qtd
    P = len(D)

    last_lambda = np.ones((K, P))
    last_prototypes = prototipos or random_prototypes(K, N, q, seed)
    last_membership_degree = None
    last_cost = None
    
    assert_relevance_weights_prod_one(last_lambda)

#     print("Passo 0")
#     print("Calculando matriz de adequação inicial (u0)")
    u0 = compute_membership_degree(weight_matrix=last_lambda,
                                   prototypes=last_prototypes,
                                   clusters_qtd=K,
                                   dissimilarity_matrices=dissimilarity_matrices,
                                   elements_qtd=N,
                                   m=m)
    
#     assert_membership_degree_sum_one(u0)

#     print("Calculando função de custo inicial (J0)")
    J0 = objective_function(clusters_qtd=K,
                            elements_qtd=N,
                            adequacy_criterion=u0,
                            m=m,
                            weight_matrix=last_lambda,
                            prototypes=last_prototypes,
                            dissimilarity_matrices=dissimilarity_matrices)
    
    last_membership_degree = u0
    last_cost = J0
    
    for t in range(1, T):
#         print(f"Passo {t}/{T}")
        
#         print(">> Calculando protótipos")
        new_prototypes = [get_prototypes(elements_qtd=N,
                                         q=q,
                                         m=m,
                                         s=1,
                                         cluster_number=k,
                                         adequacy_criterion=last_membership_degree,
                                         dissimilarity_matrices=D,
                                         weight_matrix=last_lambda) for k in range(K)]
        
        #print("new_prototypes.shape", new_prototypes)
        
#         print(">> Calculando matriz de relevâncias")
        new_lambda = compute_relevance_weights(clusters_qtd=K,
                                               dissimilarity_matrices=D,
                                               prototypes=new_prototypes,
                                               elements_qtd=N,
                                               adequacy_criterion=last_membership_degree,
                                               m=m)
        
#         assert_relevance_weights_prod_one(new_lambda)
    
#         print(">> Calculando grau de pertinência")
        new_degree = compute_membership_degree(weight_matrix=new_lambda,
                                               prototypes=new_prototypes,
                                               clusters_qtd=K,
                                               dissimilarity_matrices=dissimilarity_matrices,
                                               elements_qtd=N,
                                               m=m)
    
        
#         assert_membership_degree_sum_one(new_degree)

#         print(">> Calculando função objetivo")
        new_cost = objective_function(clusters_qtd=K,
                                      elements_qtd=N,
                                      adequacy_criterion=new_degree,
                                      m=m,
                                      weight_matrix=new_lambda,
                                      prototypes=new_prototypes,
                                      dissimilarity_matrices=dissimilarity_matrices)

        last_prototypes = new_prototypes
        last_lambda = new_lambda
        last_membership_degree = new_degree
        print(f">> Cost ({seed}): ", new_cost)
        
        if abs(last_cost - new_cost) <= epsilon:
            last_cost = new_cost
            break
    
        last_cost = new_cost
        
    data = {
        "cost":last_cost,
        "membership_degree":last_membership_degree,
        "prototypes":last_prototypes,
        "weight_matrix":last_lambda,
        "times": t,
        "q": q,
        "K":K,
        "m":m,
        "seed": seed,
    }

    return data

def carregar_matrizes_dissimiliradidades(data_path = None):
    data_path = data_path or "./data"
    data_path = os.path.abspath(data_path)
    
    FAC_FILE = os.path.join(data_path, "mfeat-fac-dissimilarity.npy")
    FOU_FILE = os.path.join(data_path, "mfeat-fou-dissimilarity.npy")
    KAR_FILE = os.path.join(data_path, "mfeat-kar-dissimilarity.npy")

    fac_dis = np.load(FAC_FILE)
    fou_dis = np.load(FOU_FILE)
    kar_dis = np.load(KAR_FILE)
    
    return fac_dis, fou_dis, kar_dis

def export_best_result(data, file_name):
    with open(file_name, "wb") as f:
        pickle.dump(data, f)

def import_best_result(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)
        
def export_fuzzy_partitions_to_csv(data, file_name):
    df = pd.DataFrame(data["membership_degree"])
    df.to_csv(file_name, index=False, decimal=',')
    

def report_run(run_number, data, report_file, name = None):
    report = {k:v for k,v in data.items() if k not in ("membership_degree", "weight_matrix")}

    if name:
      report["name"] = name

    report["run"] = run_number

    classes = clustering.get_instances_class()
    
    members, predicted_classes = clustering.get_hard_patitions(data["membership_degree"])  
    report["partition_entropy"] = clustering.calc_partition_entropy(data["membership_degree"])
    report["modified_partition_coefficient"] = clustering.calc_modified_partition_coefficient(data["membership_degree"])
    report["entropy"] = clustering.calc_partition_entropy(data["membership_degree"])
    report["adjusted_rand_score"] = adjusted_rand_score(classes, predicted_classes)
    
    report["f1_micro"] = f1_score(classes, predicted_classes, average="micro")
    report["f1_macro"] = f1_score(classes, predicted_classes, average="macro")
    report["classification_error"] = 1 - accuracy_score(classes, predicted_classes)
    
    for j, g in enumerate(members):
        report[f"g{j} size"] = len(g)
    
    
    if os.path.exists(report_file):
        pd.DataFrame([report]).to_csv(report_file, index=False, decimal=",", mode="a", header=False)
    else:
        pd.DataFrame([report]).to_csv(report_file, index=False, decimal=",", mode="a", header=True)
    
    
    
    
def executar_algoritmo_varias_vezes(dissimilarity_matrices, 
                                    N, 
                                    times=100, 
                                    name = None, 
                                    report_file=None,  
                                    **kwargs):
    best = None
    
    seeds = [18082020 + i for i in range(times)]
    #seeds = [28082020 + i for i in range(times)]
    
    with mp.Pool() as p:
        results = []
        for seed in seeds:
            kwargs = dict(kwargs)
            kwargs["seed"] = seed
            r = p.apply_async(executar_treinamento, (dissimilarity_matrices, N), kwargs)
            results.append(r)
            
        for i, r in enumerate(results):
            data = r.get()
            print(f"Execução {i+1}/{times}")
            print(">> Cost: ", data["cost"])
            if (not best) or data["cost"] < best["cost"]:
                best = data 
                
            if report_file:
                report_run(i, data, report_file, name = name)
             
        
            
    return best

def treinar_com_varios_parametros(qs, 
                                  ms, 
                                  dissimilarity_matrices, 
                                  times=10,
                                  name=None, 
                                  report_file = "data/treinamento_com_varios_paramentros.csv"):
    resultados = []

    for m, q in product(ms, qs):
        print(f">> q: {q} >> m: {m}")
        executar_algoritmo_varias_vezes(dissimilarity_matrices, 
                                        N=2000, 
                                        q=q, 
                                        m=m,
                                        name=name,
                                        report_file=report_file,
                                        times=times)


        
        
def executar_algoritmo_varias_vezes_todas(times, report_file=None, **kwargs):
    fac_dis, fou_dis, kar_dis = carregar_matrizes_dissimiliradidades()

    melhor_resultado_todas = executar_algoritmo_varias_vezes([fac_dis, fou_dis, kar_dis], 
                                                                2000, 
                                                                report_file = report_file,
                                                                times=100, **kwargs)

    export_best_result(melhor_resultado_todas, "data/melhor_resultado_todas.pickle")
    export_fuzzy_partitions_to_csv(melhor_resultado_todas, "data/fuzzy_partitions_todas.csv")

def executar_algoritmo_varias_vezes_fac(times, report_file=None, **kwargs):
    fac_dis, fou_dis, kar_dis = carregar_matrizes_dissimiliradidades()

    melhor_resultado_todas = executar_algoritmo_varias_vezes([fac_dis], 
                                                                2000, 
                                                                report_file = report_file,
                                                                times=100, **kwargs)

    export_best_result(melhor_resultado_todas, "data/melhor_resultado_fac.pickle")
    export_fuzzy_partitions_to_csv(melhor_resultado_todas, "data/fuzzy_partitions_fac.csv")

def executar_algoritmo_varias_vezes_fou(times, report_file=None, **kwargs):
    fac_dis, fou_dis, kar_dis = carregar_matrizes_dissimiliradidades()

    melhor_resultado_todas = executar_algoritmo_varias_vezes([fou_dis], 
                                                                2000, 
                                                                report_file = report_file,
                                                                times=100, **kwargs)

    export_best_result(melhor_resultado_todas, "data/melhor_resultado_fou.pickle")
    export_fuzzy_partitions_to_csv(melhor_resultado_todas, "data/fuzzy_partitions_fou.csv")

def executar_algoritmo_varias_vezes_kar(times, report_file=None, **kwargs):
    fac_dis, fou_dis, kar_dis = carregar_matrizes_dissimiliradidades()

    melhor_resultado_todas = executar_algoritmo_varias_vezes([kar_dis], 
                                                                2000, 
                                                                report_file = report_file,
                                                                times=100, **kwargs)

    export_best_result(melhor_resultado_todas, "data/melhor_resultado_kar.pickle")
    export_fuzzy_partitions_to_csv(melhor_resultado_todas, "data/fuzzy_partitions_kar.csv")


if __name__ == "__main__":
    executar_algoritmo_varias_vezes_todas(times=100, 
                                          q=2, 
                                          m=1.6, 
                                          report_file = "data/relatorio_varias_execucoes_todas.csv"
    )
    #executar_algoritmo_varias_vezes_fou(times=100, 
    #                                      q=2, 
    #                                      m=1.1, 
    #                                      report_file = "data/relatorio_varias_execucoes_kar.csv"
    #)

    sys.exit()

    fac_dis, fou_dis, kar_dis = carregar_matrizes_dissimiliradidades()
    #qs = list(range(2, 6))
    qs = [2]
    ms = np.arange(1.2, 1.5, .1)


    treinar_com_varios_parametros(qs, 
                                  ms, 
                                  [fac_dis, fou_dis, kar_dis], 
                                  times=100,
                                  name="todas", 
                                  report_file = "data/treinamento_com_varios_paramentros.csv")

    treinar_com_varios_parametros(qs, 
                                  ms, 
                                  [fac_dis], 
                                  times=100,
                                  name="fac", 
                                  report_file = "data/treinamento_com_varios_paramentros.csv")

    treinar_com_varios_parametros(qs, 
                                  ms, 
                                  [fou_dis], 
                                  times=100,
                                  name="fou", 
                                  report_file = "data/treinamento_com_varios_paramentros.csv")

    treinar_com_varios_parametros(qs, 
                                  ms, 
                                  [kar_dis], 
                                  times=100,
                                  name="kar", 
                                  report_file = "data/treinamento_com_varios_paramentros.csv")

#     # fuzzy.buscar_melhores_parametros(qs, ms, [fac_dis, fou_dis, kar_dis])
#     print(">> FAC")
#     buscar_melhores_parametros(qs, ms, [fac_dis], file_name="data/melhores_parametros_fac.csv", times= 10)
#     print(">> FOU")
#     buscar_melhores_parametros(qs, ms, [fou_dis], file_name="data/melhores_parametros_fou.csv", times = 10)
#     print(">> DIS")
#     buscar_melhores_parametros(qs, ms, [kar_dis], file_name="data/melhores_parametros_kar.csv", times = 10)