#!/usr/bin/env python
# coding: utf-8

# # Implementação do algoritimo completo, sem compromisso com  perfomance

# ## Imports

# In[2]:


import os
import functools
import operator
import math
import random
import sys

import numpy as np


# ## Função match
# 
# ![Match function](./img/match_function.png)

# In[3]:


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

    # Criando aliases compatíveis com os nomes da fórmula
    k = cluster_number
    D = dissimilarity_matrices
    p = len(D)
    Gk = prototypes[k]

#     dissimilarities_sum = np.array(
#         [sum([dj[element, e] for e in Gk]) for dj in D])

    dissimilarities_sum = np.array([dj[element, Gk].sum() for dj in D])

    return np.dot(weight_matrix[k], dissimilarities_sum)


# ## Função objetivo
# 
# ![Objetive function](./img/objective_function.png)

# In[4]:


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
    Gk = prototypes
    N = elements_qtd
    match = cluster_matching_function # Criando um alias para reduzir o nome da função de matching
  
    J = [sum([u[i, k] * match(l, k, i, Gk, D) for i in range(N)]) 
          for k in range(K)]


    return sum(J)


# ## Cálculo de protótipos
# 
# ![Prototype function](./img/prototype_function.png)

# In[5]:


def get_prototypes(elements_qtd,
                   q,
                   m,
                   s,
                   cluster_number,
                   adequacy_criterion,
                   dissimilarity_matrices,
                   weight_matrix):
    """
        :params:
                elements_qtd: int 
                    Quantidade de elementos da base de dados
                q: int
                    Quantidade de elementos protótipos
                m: int
                    Fator de ponderação do índice de adequação
                s: int
                    Fator de ponderação dos pesos das matrizes
                cluster_number: int
                    Quantidade total de clusters
                adequacy_criterion: numpy array-like
                    Matriz u de tamanho N x K contendo a índice de adequação 
                    de cada elemente a cada cluster
                dissimilarity_matrices: lista de numpy array
                    Lista de matrizes de dissimilaridade
                weight_matrix: 
                     matriz K x P de pesos das matrizes de dissimilaridades por cluster

        :return: list

    """

    G = []
    k = cluster_number,
    D = dissimilarity_matrices
    u = np.power(adequacy_criterion, m)
    l = np.power(weight_matrix, s)
    N = elements_qtd
    p = len(D)

    def dist(element):
        """
            Função auxiliar para cálculo da distância de um elemento qualquer 
            em relação a todos os outros da base de dados, consideran as matrizes 
            de dissimilaridade e 
            o critério de adequação

            return: (int, float)
                (lement, soma das distâncias)
        """
        return element, sum([u[i, k] * sum([l[k, j] * D[j][element, i] for j in range(p)])
                             for i in range(elements_qtd)])

    while (len(G) < q):
        # Calculando todas as distâncias dos elementos que ainda não estão em G
        all_distances = (dist(i) for i in range(N) if i not in G)
        # Obtendo o menor item de distância e separando entre o elemento e sua distância
        element, _ = min(all_distances, key=operator.itemgetter(1))
        G.append(element)

    return G


# ## Matriz de relevâcia
# 
# ![Funções de peso](./img/vector_weights_function.png)

# In[6]:


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

        :return: numpy array

    """

    D = dissimilarity_matrices
    P = len(D)
    Gk = prototypes
    K = clusters_qtd
    N = elements_qtd
    u = np.power(adequacy_criterion, m)
    l = np.zeros((K, P))

    def match(element, Dh, G):
        """
            Função auxiliar para cálculo de match entre um elemento 
            qualquer, os protótipos G de um cluster específico e uma matriz 
            de similaridade específica Dh.
        """

        return sum([Dh[element, e] for e in G])

    for k in range(K):
        for j in range(P):
            # Calculado o somatório do numerador da equação à esquerda da igualdade
            weight_diss_sum1 = [sum([u[i, k] * match(i, D[h], Gk[k]) for i in range(N)])
                                for h in range(P)]

            # Calculando o produtório das somas acima estabelecidas
            weight_diss_prod = functools.reduce(
                operator.mul, weight_diss_sum1)

            # Calculado o somatório do denominador da equação à esquerda da igualdade
            weight_diss_sum2 = sum([u[i, k] * match(i, D[j], Gk[k])
                                    for i in range(N)])

            # Executando a divisão da fração à esquerda da equação
            l[k, j] = math.pow(weight_diss_prod, 1/P) / weight_diss_sum2

    return l


# ## Grau de pertinência
# 
# ![Fórmula grau de pertinência](./img/membership_degree.png)

# In[7]:


def compute_membership_degree(weight_matrix,
                              prototypes,
                              clusters_qtd,
                              dissimilarity_matrices,
                              elements_qtd,
                              m):

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
        return math.pow(r, 1/(m-1))

    for i in range(N):
        for k in range(K):
            outter_sum = sum([ratio(i, k, h) for h in range(K)])
            u[i, k] = 1/outter_sum

    return u


# ## Algoritmo completo
# > Partitioning fuzzy K-medoids clustering algorithms with relevance weight for each dissimilarity matrix estimated locally
# 
# * Parametros: $K = 10; m = 1.6; T = 150; \epsilon = 10^{−10};$
# * Devemos considerar a iniciarlizar do vetor de pesos como sendo 1, já que usamos a equação 9 (MFCMdd-RWL-P)

# In[8]:


def random_prototypes(K, N, q):
    elements = set(range(N))
    protos = []
    for k in range(K):
        protos.append(random.sample(elements, q))
        elements -= set(protos[-1])

    return protos


def executar_algoritmo(dissimilarity_matrices,
                       elements_qtd,
                       K=10,
                       m=1.6,
                       T=150,
                       epsilon=10e-10,
                       q=3):

    D = dissimilarity_matrices
    N = elements_qtd
    P = len(D)

    last_lambda = np.ones((K, P))
    last_prototypes = random_prototypes(K, N, q)
    last_membership_degree = None
    last_cost = None

    print("Passo 0")
    print("Calculando matriz de adequação inicial (u0)")
    u0 = compute_membership_degree(weight_matrix=last_lambda,
                                   prototypes=last_prototypes,
                                   clusters_qtd=K,
                                   dissimilarity_matrices=dissimilarity_matrices,
                                   elements_qtd=N,
                                   m=m)

    print("Calculando função de custo inicial (J0)")
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
        print(f"Passo {t}/{T}")
        new_prototypes = [get_prototypes(elements_qtd=N,
                                         q=q,
                                         m=m,
                                         s=1,
                                         cluster_number=k,
                                         adequacy_criterion=last_membership_degree,
                                         dissimilarity_matrices=D,
                                         weight_matrix=last_lambda) for k in range(K)]

        new_lambda = compute_relevance_weights(clusters_qtd=K,
                                               dissimilarity_matrices=D,
                                               prototypes=new_prototypes,
                                               elements_qtd=N,
                                               adequacy_criterion=last_membership_degree,
                                               m=m)

        new_degree = compute_membership_degree(weight_matrix=new_lambda,
                                               prototypes=new_prototypes,
                                               clusters_qtd=K,
                                               dissimilarity_matrices=dissimilarity_matrices,
                                               elements_qtd=N,
                                               m=m)

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
        
        if abs(last_cost - new_cost) <= epsilon:
            last_cost = new_cost
            break
    
        last_cost = new_cost
        
    data = {
        "cost":last_cost,
        "membership_degree":last_membership_degree,
        "prototypes":last_prototypes,
        "weight_matrix":last_lambda 
    }
    return data


# ## Carregando Matrizes

# In[13]:


def carregar_matrizes(data_path = None):
    data_path = data_path or "./data"
    data_path = os.path.abspath(data_path)
    print(data_path)
    
    FAC_FILE = os.path.join(data_path, "mfeat-fac-dissimilarity.npy")
    FOU_FILE = os.path.join(data_path, "mfeat-fou-dissimilarity.npy")
    KAR_FILE = os.path.join(data_path, "mfeat-kar-dissimilarity.npy")

    fac_dis = np.load(FAC_FILE)
    fou_dis = np.load(FOU_FILE)
    kar_dis = np.load(KAR_FILE)
    
    return fac_dis, fou_dis, kar_dis


# ## Executando algoritmo

# In[14]:


def main():
    data_path = sys.argv[1] if len(sys.argv) == 2 else None
    
    fac_dis, fou_dis, kar_dis = carregar_matrizes(data_path)
    
    dissimilarity_matrices = [fac_dis, fou_dis, kar_dis]
    N = fac_dis.shape[0]
    data = executar_algoritmo(dissimilarity_matrices, N)
    
    print(data)

main()


# In[ ]:




