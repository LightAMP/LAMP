

# -*- coding: utf-8 -*-
"""
corrupted Mixes
"""

import numpy as np
import itertools


import numpy as np
from sklearn_extra.cluster import KMedoids
import random

def kmedoids_clustering(data, num_clusters):
    # Extract latitude and longitude values
    coordinates = np.array([[float(point["latitude"]), float(point["longitude"])] for point in data])

    # Perform k-medoids clustering
    kmedoids = KMedoids(n_clusters=num_clusters, random_state=42)
    clusters = kmedoids.fit_predict(coordinates)

    return clusters

def find_closest_points(data, m):
    num_clusters = 3
    if m > len(data):
        raise ValueError("m should be less than or equal to the length of the data list")

    # Perform k-medoids clustering
    clusters = kmedoids_clustering(data, num_clusters)

    # Find the cluster with the most nodes
    largest_cluster = max(set(clusters), key=clusters.tolist().count)


    # Select m nodes randomly from the largest cluster
    indices_of_selected_nodes = [i for i, cluster_label in enumerate(clusters) if cluster_label == largest_cluster]
    selected_indices = random.sample(indices_of_selected_nodes, min(m, len(indices_of_selected_nodes)))

    # Get selected nodes based on indices
    selected_nodes = [data[i] for i in selected_indices]

    return selected_nodes, selected_indices




def findindices(prob_dist, s):
    n = len(prob_dist)
    
    # Check if s is greater than the length of the probability distribution
    if s > n:
        raise ValueError("s cannot be greater than the length of the probability distribution")

    # Generate all combinations of indices for subsets of size s
    index_combinations = list(itertools.combinations(range(n), s))

    # Calculate the sum of probabilities for each combination
    subset_sums = [sum(prob_dist[i] for i in indices) for indices in index_combinations]

    # Find the indices with the maximum sum
    max_sum_indices = max(enumerate(subset_sums), key=lambda x: x[1])[0]

    # Return the indices with the maximum sum
    result_indices = index_combinations[max_sum_indices]

    return result_indices


def nCr(n,r):
    import math
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def Path_Fraction(a,b,c,Dict,W):
    Term = 0
    if len(a) !=0 and len(b) !=0 and len(c) !=0:
        for item1 in a:
            for item2 in b:
                for item3 in c:
                    Term = Term + (1/W)*(Dict['PM%d' %item1][item2-W-1])*(Dict['PM%d' %item2][item3-2*W-1])
    return Term

def sort_index(List , X):
    x = 0
    INDEX = []
    while(x < X):
        Max = max(List)
        Indx = List.index(Max)
        INDEX.append(Indx)
        List[Indx] = 0
        x = x+1
    return INDEX


def sort_of_clusters(Labels1):
    lists = Labels1
    Index1 = []
    for i in range(len(lists)):
        maxs = 0
        j = 0
        for item in lists:
            if item > maxs:
                maxs = item
                index1 = j
            j = j +1
        lists[index1] = 0
        Index1.append(index1)
    return Index1
        
class  FCP_Mix(object):
    def __init__(self,Data,Budget):
        self.Hyper = 10
        self.N = int((3/2)*len(Data))
        self.W = int(self.N/3)
        self.Data = Data
        self.CNodes = int(self.N*Budget/100)
        self.Budget = Budget/100

        
 
            
            
            
    def C_random(self,G_mean):
        #print(G_mean)
        Dict = self.Data
        CNodes = {}
        item    = []
        Index_x = []
        Index_y = []
        for i in range(self.N):
            coin = np.random.multinomial(1, [self.Budget,1-self.Budget], size=1)[0][0]
            if coin == 1:
                if i<self.W:
                    item.append(i)
                elif i>2*self.W-1:
                    Index_y.append(i-2*self.W)
                else:
                    Index_x.append(i-self.W)
                    
                j = i +1
                CNodes['PM%d' %j] = True
            else:
                j = i +1
                CNodes['PM%d' %j] = False
       # print('hi')
        #print(item,Index_x,Index_y)    
 
        Term1 = 0
        for item1 in item:
            for item2 in Index_x:
                for item3 in Index_y:
                    Term1 = Term1 + (G_mean[item1])*(Dict['PM%d' %(item1+1)][item2])*(Dict['PM%d' %(self.W+1+item2)][item3])
        return CNodes,Term1
            
            
            
 
            
    def Close_knit_nodes(self,G_mean,data):
        if len(data)>self.W:
            Selected_Dicts,Index_Dict = find_closest_points(data,self.CNodes)
        else:
            Index_Dict = []
            counter = 0
            jj = 0
            Index_Dict1 = []            
            Index_Dict2 = []
            Index_Dict3 = []
            
            for element in range(len(data)):
                W_now = int(len(data[element])/3)
                Selected_Dicts,Index_Dict_ = find_closest_points(data[element],int(self.Budget*(3*W_now)))

                for part in Index_Dict_:
                    if part < W_now:
                        Index_Dict1.append(part+counter)
                    elif  (W_now-1)<part < 2*W_now:
                        Index_Dict2.append(part-W_now+counter)
                    elif part > 2*W_now-1:
                        Index_Dict3.append(part-2*W_now+counter)
                counter = counter + W_now
            for i in range(len(Index_Dict1)):
                Index_Dict.append(Index_Dict1[i])
                Index_Dict.append(Index_Dict1[i]+counter)
                Index_Dict.append(Index_Dict1[i]+2*counter)
        
            self.W = counter
        Dict = self.Data
        Dict_ = {}
        for j in range(self.N):
            Dict_['PM'+str(j+1)] = False
        item    = []
        Index_x = []
        Index_y = []
        
        for i in range(self.N):
            if i in Index_Dict:
                Dict_['PM'+str(i+1)] = True

                if i<self.W:
                    item.append(i)
                elif i>2*self.W-1:
                    Index_y.append(i-2*self.W)
                else:
                    Index_x.append(i-self.W)            
                
        Term1 = 0
        for item1 in item:
            for item2 in Index_x:
                for item3 in Index_y:
                    Term1 = Term1 + (G_mean[item1])*(Dict['PM%d' %(item1+1)][item2])*(Dict['PM%d' %(self.W+1+item2)][item3])
        return Dict_,Term1
    
    
    def Greedy_For_Fairness(self,G_mean):
        import numpy as np
        #Initially we consider all the nodes as hones one
        CNodes = {}
        for i in range(self.N):
            j = i +1
            CNodes['PM%d' %j] = False  
        C_i = int(self.CNodes/3)
        
        
        Dict = self.Data

        Indicator = False #Shows the load balanced network or atleast the first layer
        #for element in G_mean:#if node selection from forst layer is not uniform then it's not 
            #balanced
          #  if not element==1/self.W:
           #     Indicator = True
            #    break
        if  Indicator :
            W_List = findindices(G_mean,int(self.CNodes/3)) 
        
        else:
            import random
            WL = []
            LIs = [j+1 for j in range(self.W)]
       
            while len(WL)<self.W*self.Hyper:

                RNDM = tuple(random.sample(LIs, C_i))
                if not (RNDM in WL):
                    WL.append(RNDM)
            WL = set(WL)            
            
            

        
     
        if Indicator:

            c = 3*C_i           
            item = list(W_List)
            X = np.zeros((C_i,self.W))
            j = 0
            for terms in item:
                X[j,:] = Dict['PM%d' %(terms+1)]
                j = j+1
            X_SUM = np.sum(X , axis = 0)
            x_sum = X_SUM.tolist()

            Index_x = sort_index(x_sum , C_i)

            Y = np.zeros((C_i,self.W))
            j = 0
            for terms in Index_x:
                Y[j,:] = Dict['PM%d' %(terms+1+self.W)]
                j = j +1

            Y_SUM = np.sum(Y , axis = 0)
            y_sum = Y_SUM.tolist()

            Index_y = sort_index(y_sum,C_i)

            
            while(c < self.CNodes):

                Par = -1
                for m in range(0,self.W):
                    if not (m in item ):
                        parameter = 0
                        for TRm in Index_x:
                            parameter = parameter + Dict['PM%d' %(m+1)][int(TRm)]
                        if Par < parameter:
                            Par = parameter
                            Inx = m
                            
                
                for m in range(self.W+1,2*self.W+1):
                    if not ((m-1-self.W) in Index_x ):
                        parameter = 0
                        for TRm1 in Index_y:
                            parameter = parameter + 0.5*Dict['PM%d' %(m)][int(TRm1)]
                        for TRm2 in item:
                            parameter = parameter + 0.5*Dict['PM%d' %(int(TRm2+1))][m-1-self.W]
                        if Par < parameter:
                            Par = parameter
                            Inx = m
            
                
                for m in range(2*self.W+1,3*self.W+1):
                    if not ((m-1-2*self.W) in Index_y ):
                        parameter = 0
                        for TRm in Index_x:
                            parameter = parameter + Dict['PM%d' %(int(TRm)+self.W+1)][m-2*self.W-1]
                        if Par < parameter:
                            Par = parameter
                            Inx = m
                if Inx < self.W+1:
                    item.append(Inx)
                elif self.W< Inx <2*self.W+1:
                    Index_x.append(Inx -1 -self.W)
                elif 2*self.W < Inx:
                    Index_y.append(Inx-2*self.W-1)
                c = c+1

            Term1 = 0
            for item1 in item:
                for item2 in Index_x:
                    for item3 in Index_y:
                        Term1 = Term1 + (G_mean[item1])*(Dict['PM%d' %(item1+1)][item2])*(Dict['PM%d' %(self.W+1+item2)][item3])
            A = item
            B = Index_x
            C = Index_y
            Max = Term1
                    
                    
                    
            for i in range(self.N):
                if not ((i) in A) and not ((i-self.W) in B) and not ((i-2*self.W) in C):
                    j = i +1
                    CNodes['PM%d' %j] = False
                else:
                    j = i +1
                    CNodes['PM%d' %j] = True                     
        else:
                            
            Max = 0
            for itemm in WL:
                c = 3*C_i           
                item = list(itemm)
                X = np.zeros((C_i,self.W))
                j = 0
                for terms in item:
                    X[j,:] = Dict['PM%d' %terms]
                    j = j+1
                X_SUM = np.sum(X , axis = 0)
                x_sum = X_SUM.tolist()
    
                Index_x = sort_index(x_sum , C_i)
    
                Y = np.zeros((C_i,self.W))
                j = 0
                for terms in Index_x:
                    Y[j,:] = Dict['PM%d' %(terms+1+self.W)]
                    j = j +1
    
                Y_SUM = np.sum(Y , axis = 0)
                y_sum = Y_SUM.tolist()
    
                Index_y = sort_index(y_sum,C_i)
    
                
                while(c < self.CNodes):
    
                    Par = -1
                    for m in range(1,self.W+1):
                        if not (m in item ):
                            parameter = 0
                            for TRm in Index_x:
                                parameter = parameter + Dict['PM%d' %(m)][int(TRm)]
                            if Par < parameter:
                                Par = parameter
                                Inx = m
                                
                    
                    for m in range(self.W+1,2*self.W+1):
                        if not ((m-1-self.W) in Index_x ):
                            parameter = 0
                            for TRm1 in Index_y:
                                parameter = parameter + 0.5*Dict['PM%d' %(m)][int(TRm1)]
                            for TRm2 in item:
                                parameter = parameter + 0.5*Dict['PM%d' %(int(TRm2))][m-1-self.W]
                            if Par < parameter:
                                Par = parameter
                                Inx = m
                
                    
                    for m in range(2*self.W+1,3*self.W+1):
                        if not ((m-1-2*self.W) in Index_y ):
                            parameter = 0
                            for TRm in Index_x:
                                parameter = parameter + Dict['PM%d' %(int(TRm)+self.W+1)][m-2*self.W-1]
                            if Par < parameter:
                                Par = parameter
                                Inx = m
                    if Inx < self.W+1:
                        item.append(Inx)
                    elif self.W< Inx <2*self.W+1:
                        Index_x.append(Inx -1 -self.W)
                    elif 2*self.W < Inx:
                        Index_y.append(Inx-2*self.W-1)
                    c = c+1
    
                Term1 = 0
                for item1 in item:
                    for item2 in Index_x:
                        for item3 in Index_y:
                            Term1 = Term1 + (1/self.W)*(Dict['PM%d' %item1][item2])*(Dict['PM%d' %(self.W+1+item2)][item3])
                if Max<Term1:
                    Max = Term1
                    A = item
                    B = Index_x
                    C = Index_y
                    
                    
                    
            for i in range(self.N):
                if not ((i+1) in A) and not ((i-self.W) in B) and not ((i-2*self.W) in C):
                    j = i +1
                    CNodes['PM%d' %j] = False
                else:
                    j = i +1
                    CNodes['PM%d' %j] = True                

        return CNodes,Max
    
    
    '''    
#Test of Function       
import numpy as np

W = 20
data = {}
for i in range(2*W):
    a = np.random.rand(1, W)
    b = [item / sum(a[0]) for item in a[0]]
    data['PM'+str(i+1)] = b
    
    

a = np.random.rand(1, W)
b = [item / sum(a[0]) for item in a[0]]

G_mean = b    
G_Mean = [1/W for i in range(W)]
        
        
print(G_mean)
        
C = FCP_Mix(data,30)
 
D,E = C.Greedy_For_Fairness(G_mean)
print(D,E)



D,E = C.C_random(G_mean)





print(E,D)
import json

with open('D:/Approach3/ripe_November_12_2023_cleaned.json') as file:
    Data0 = json.load(file)

xx = [Data0[0:15],Data0[15:45],Data0[45:60]]
D,E = C.Close_knit_nodes(G_mean, Data0[0:60])

i = 0
for item in D:
    if D[item]:
        i = i+1
print(i)

D,E = C.Close_knit_nodes(G_mean, xx)

i = 0
for item in D:
    if D[item]:
        i = i+1
print(i)

print(E,D)

'''


'''  
a,b = find_closest_points(Data0,10)

print(b)
'''