#E2E# -*- coding: utf-8 -*-
"""
Circle
"""
'''
import numpy as np

a = np.matrix([[1,2,3],[3,-5,6],[6,-7,8]])
b= [1,2,3]
c = np.abs(a-np.abs(np.matrix(b)))
D = np.sum(c,axis = 1)
print(min(D.tolist()))
'''
'''
        
List = [12,14,15,16,17,18,19,20,30,30,40,41,42,42,43,44,45,46,47,45,45,45,45,45,45,80] 

print((max(List)-min(List))/5)
'''
from math import exp
from scipy import constants

# Import library for making the simulation, making random choices,
#creating exponential delays, and defining matrixes.
from scipy.stats import expon
import simpy
import random
import numpy  as np
import pickle

from Message_ import message

from GateWay import GateWay
from Mix_Node_ import Mix

from NYM import MixNet

from Message_Genartion_and_mix_net_processing_ import Message_Genartion_and_mix_net_processing

import json
import numpy as np
def I_key_finder(x,y,z,matrix,data):
    
    List = [x,y,z]
    Index1 = np.sum(np.abs(matrix - List),axis = 1)
    index = Index1.tolist()
    Index2 = min(index)
    Index = index.index(Index2)

    
    return data[Index]
            
def Medd(List):
    N = len(List)

    List_ = []
    import statistics
    for i in range(N):

        List_.append( statistics.median(List[i]))
        
    return List_


def Loc_finder(I_key,data):
    for i in range(len(data)):
        if data[i]['i_key'] == I_key:
            return i
    
def Ent(List):
    L =[]
    for item in List:
       
        if item!=0:
            L.append(item)
    l = sum(L)
    for i in range(len(L)):
        L[i]=L[i]/l
    ent = 0
    for item in L:
        ent = ent - item*(np.log(item)/np.log(2))
    return ent

def Med(List,Per):
    import numpy as np
    N = len(List)

    List_ = []
    import statistics
    for i in range(N):

        List_.append( np.percentile(List[i], Per))
        
    return List_


from itertools import combinations
from math import radians, sin, cos, sqrt, atan2






class CircularMixNet_MC(object):
    
    def __init__(self,Targets,Iteration,Capacity,run,delay1,delay2,H_N,N,rate,num_gateways,Percentile,Goal):
        self.Iterations = Iteration
        self.CAP = Capacity
        self.rate = rate
        self.d1 = delay1
        self.d2 = delay2
        self.H_N = H_N
        self.N_target = Targets
        self.N = N
        self.W = round(N/3)
        self.G = num_gateways
        self.run = run
        self.Goal = Goal
        self.CN = 1
        self.c_node = int(0.20*self.N)
        self.data_type = 'RIPE'
        self.PP = Percentile
        self.Adversary_Budget = 30
        self.Var =  [0.0,0.001,0.005,0.01,0.05,0.1]

    def haversine_distance(self,lat1, lon1, lat2, lon2):
        """
        Calculate the haversine distance between two points specified by latitude and longitude.
        """
        R = 6371  # Radius of the Earth in kilometers
    
        # Convert latitude and longitude from degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
    
        return distance
    
    def find_closest_subset(self,data, subset_size):
        """
        Find the subset of dictionaries that are closest to each other based on latitude and longitude.
        """
        n = len(data)
    
        if n < subset_size:
            raise ValueError("Number of dictionaries is less than the specified subset size.")
    
        # Initialize variables to store the minimum distance and corresponding subset
        min_distance = float('inf')
        closest_subset = []
    
        # Iterate over all possible combinations of subset_size dictionaries
        for subset_indices in combinations(range(n), subset_size):
            subset_distance = sum(self.haversine_distance(data[i]['latitude'], data[i]['longitude'],
                                                      data[j]['latitude'], data[j]['longitude'])
                                 for i, j in combinations(subset_indices, 2))
    
            # Check if the current subset has a smaller total distance
            if subset_distance < min_distance:
                min_distance = subset_distance
                closest_subset = subset_indices
    
        return closest_subset



    def make_data(self):
        from Datasets_ import Dataset
        DF  = Dataset(self.data_type,self.N,self.Goal,self.CN,self.G)        
        DF.RIPE()        
        Data_set, Client_Data , GW_Data = DF.PLOT_New_dataset()
        self.Data_set = Data_set
        self.Clients  = Client_Data
        self.GW_Data      = GW_Data

    def MixNet_Creation(self):  
        GateWays = {}
        Layer = {'Layer1':{},'Layer2':{}}
        import json
        import numpy as np
    
        with open('ripe_November_12_2023_cleaned.json') as json_file: 
        
            data0 = json.load(json_file) 
        number_of_data = len(data0)-1
        List = []
        i=0
        while(i<4*self.W):
            a = int(number_of_data*np.random.rand(1)[0]+1)
            if a > number_of_data:
                a==number_of_data
            if not a in List:
                List.append(a)
                i = i +1
                
        B = [ data0[item]  for item in List if List.index(item) > (self.W-1)]

        
        self.close_data =   B
                

        for i in range(self.W):
            ID1 = List[i]
            for j in range(self.W,2*self.W):
                ID2 = List[j] 
                I_key = data0[ID2]['i_key']
                In_Latency = data0[ID1]['latency_measurements'][str(I_key)]
                delay_distance = int(In_Latency)
                if delay_distance == 0:
                    delay_distance =1
                GateWays['G'+str(i+1) +'PM'+str(1+j-self.W)] = abs(delay_distance)/2000
            

        for k in range(2):
            for i in range(k*self.W,(k+1)*self.W):
                ID1 = List[i+self.W]
                for j in range((k+1)*self.W,(k+2)*self.W):
                    ID2 = List[j+self.W]   
                    I_key = data0[ID2]['i_key']
                    In_Latency = data0[ID1]['latency_measurements'][str(I_key)]
                    delay_distance = int(In_Latency)
                    if delay_distance == 0:
                        delay_distance =1                    
                    Layer['Layer'+str(k+1)]['PM'+str(i+1) +'PM'+str(j+1)]= abs(delay_distance)/2000                 
                    
        return GateWays,Layer            
            
            



    def Large_MixNet_Creation(self):  
        GateWays = {}
        Layer = {'Layer1':{},'Layer2':{}}
        import json
        import numpy as np
    
        with open('ripe_November_12_2023_cleaned.json') as json_file: 
        
            data0 = json.load(json_file) 
        number_of_data = len(data0)-1
        List = []
        i=0
        while(i<4*self.W):
            a = int(number_of_data*np.random.rand(1)[0]+1)
            if a > number_of_data:
                a==number_of_data
            if not a in List:
                List.append(a)
                i = i +1
                
        B = [ data0[item]  for item in List if List.index(item) > (self.W-1)]

        
        self.close_data =   B
                

        for i in range(self.W):
            ID1 = List[i]
            for j in range(self.W,2*self.W):
                ID2 = List[j] 
                I_key = data0[ID2]['i_key']
                In_Latency = data0[ID1]['latency_measurements'][str(I_key)]
                delay_distance = int(In_Latency)
                if delay_distance == 0:
                    delay_distance =1
                GateWays['G'+str(i+1) +'PM'+str(1+j-self.W)] = abs(delay_distance)/2000
            

        for k in range(2):
            for i in range(k*self.W,(k+1)*self.W):
                ID1 = List[i+self.W]
                for j in range((k+1)*self.W,(k+2)*self.W):
                    ID2 = List[j+self.W]   
                    I_key = data0[ID2]['i_key']
                    In_Latency = data0[ID1]['latency_measurements'][str(I_key)]
                    delay_distance = int(In_Latency)
                    if delay_distance == 0:
                        delay_distance =1                    
                    Layer['Layer'+str(k+1)]['PM'+str(i+1) +'PM'+str(j+1)]= abs(delay_distance)/2000                 
                    
        return GateWays,Layer            
            











    def MixNet_Creation_(self,Iteration):  
        import json
        with open('LARMIX__.json','r') as file:
            data0 = json.load(file)
        self.close_data = []    
        GateWays = {}
        GateWays_,Layer_,C_D = data0['Iteration'+str(Iteration+1)]
        Layer = {'Layer1':{},'Layer2':{}}
        n = int(len(C_D)*(1/3))
        for k in range(3):
            for i in range(n*k,(n)*(k+1)):
                if k*n <= i < k*n+self.W:
                    self.close_data.append(C_D[i])
                    


        
        for i in range(self.W):
            for j in range(self.W):
            
                GateWays['G'+str(i+1) +'PM'+str(1+j)] = GateWays_['G'+str(i+1) +'PM'+str(1+j)]
                
        #print(Layer_['Layer1'])
        for k in range(2):
            for i in range(k*n,k*n+self.W):
                for j in range((k+1)*n,(k+1)*n+self.W):
                    Layer['Layer'+str(k+1)]['PM'+str(i+1-k*n+k*self.W) +'PM'+str(j+1-(k+1)*n+(k+1)*self.W)] = Layer_['Layer'+str(k+1)]['PM'+str(i+1) +'PM'+str(j+1)]


        return GateWays,Layer    

    

    
    def Circles_Creation(self,Iterations,Common = False): 
        self.Var =  [0.0,0.001,0.005,0.01,0.05,0.1]
        import numpy as np

        DATA = {}
        for I in range(Iterations):
            if Common:
                GateWays,Layers = self.MixNet_Creation_(I)
            
            else:
                GateWays,Layers = self.MixNet_Creation()
            

            Policies = {}
            Policies['Close'] = self.close_data
            for j in range(self.G):
                Ascenading_sort = []
                Sort_Routs = [0]*self.W
                for i in range(self.W):
                    Sort_Routs[i] = GateWays['G'+str(j+1)+'PM'+str(i+1)]
                MATRIX = np.matrix(Sort_Routs)   
                Copy_List = Sort_Routs

                
                INDEX = []
                for k in range(self.W):
                    INDEX.append(Copy_List.index(min(Copy_List)))
                    Copy_List[Copy_List.index(min(Copy_List))] = 100000
                Policies['G'+str(j)] = [MATRIX.tolist()[0],INDEX]
            for J in range(2):
                
                for j in range(self.W):
                    Sort_Routs = [0]*self.W
                    for i in range(self.W):
                        Sort_Routs[i] = Layers['Layer'+str(J+1)]['PM'+str(J*self.W+j+1)+'PM'+str((J+1)*self.W+i+1)]     
                    MATRIX = np.matrix(Sort_Routs)
                    Copy_List = Sort_Routs
                    INDEX = []
                    for k in range(self.W):
                        INDEX.append(Copy_List.index(min(Copy_List)))
                        Copy_List[Copy_List.index(min(Copy_List))] = 100000
                    Policies['PM'+str(J*self.W+j+1)] = [MATRIX.tolist()[0],INDEX]  
                
                
                
            DATA['Iteration'+str(I)] = Policies
            
        return DATA        
    def sort_and_get_mapping(self,initial_list):
        # Sort the initial list in ascending order and get the sorted indices
        sorted_indices = sorted(range(len(initial_list)), key=lambda x: initial_list[x])
        sorted_list = [initial_list[i] for i in sorted_indices]
    
        # Create a mapping from sorted index to original index
        mapping = {sorted_index: original_index for original_index, sorted_index in enumerate(sorted_indices)}
    
        return sorted_list, mapping
    
    def restore_original_list(self,sorted_list, mapping):
        # Create the original list by mapping each element back to its original position
        original_list = [sorted_list[mapping[i]] for i in range(len(sorted_list))]
        
        return original_list
    def LARMIX(self,LIST_,Tau):#We materealize our function for making the trade off
        #In this function just for one sorted distribution
        t = Tau
        A, mapping = self.sort_and_get_mapping(LIST_)
        T = 1-t
    
        import math
        B=[]
        D=[]
    
    
        r = 1
        for i in range(len(A)):
            j = i
            J = (j*(1/(t**(r))))**(1-t)
    
            E = math.exp(-1)
            R = E**J
    
            B.append(R)
            A[i] = A[i]**(-T)
    
            g = A[i]*B[i]
    
            D.append(g)
        n=sum(D)
        for l in range(len(D)):
            D[l]=D[l]/n
        restored_list = self.restore_original_list(D, mapping)
    
        return restored_list   
    def PDF(self,LIST_Delay,Index,Bias=False,LARMIX_=False):
        #print(LIST_Delay)
        Dist = [0]*len(LIST_Delay)
        if not Bias and not LARMIX_:
            n = len(Index)
            for item in Index:
                Dist[item] = 1/n
        elif Bias and not LARMIX_:
            s = 0
            for item in Index:
                s = s+1/LIST_Delay[item]
            for item in Index:
                Dist[item] = (1/LIST_Delay[item])/s
                
        elif Bias and LARMIX_:
            LIST = []
            for item in Index:
                LIST.append(LIST_Delay[item])
            Dist_ = self.LARMIX(LIST,0.6)
            i = 0
            for item in Index:
                Dist[item]=Dist_[i]
                i = i+1
        return Dist        


    def PDF__(self,LIST_Delay,Index,tau):
        Dist = [0]*len(LIST_Delay)

        LIST = []
        for item in Index:
            LIST.append(LIST_Delay[item])
        Dist_ = self.LARMIX(LIST,tau)
        i = 0
        for item in Index:
            Dist[item]=Dist_[i]
            i = i+1
        return Dist     











    def filter_matrix_entries(self,matrix, threshold):
        
        import numpy as np
        # Convert the matrix to numpy array for easier manipulation
        matrix = np.array(matrix)
        
        # Boolean indexing to filter entries greater than the specified value
        filtered_entries = matrix[matrix > threshold]
        
        # Convert the filtered entries to a list
        filtered_list = filtered_entries.tolist()
        
        return filtered_list




       
    def compatible(self,A,B):
        a = len(A)
        b = len(B)
        X = []
    
        Min = min(a,b)
        if not a > Min:
            i = 0
            for item in A:
            
                X.append(B[i])
                i =i +1
            return A,X
        else:
            i = 0
            for item in B:
                X.append(A[i])
                i = i+1
            return X,B            
            
    

    def Simulator(self,corrupted_Mix,Mix_Dict): 
        import simpy
        Mixes = [] #All mix nodes
        GateWays = {}
        env = simpy.Environment()    #simpy environment
        capacity=[]
        for j in range(self.N):# Generating capacities for mix nodes  
            c = simpy.Resource(env,capacity = self.CAP)
            capacity.append(c)           
        for i in range(self.N):#Generate enough instantiation of mix nodes  
            ll = i +1
            X = corrupted_Mix['PM%d' %ll]
            x = Mix(env,'M%02d' %i,capacity[i],X,self.N_target,self.d1)
            Mixes.append(x)
        
 
        for i in range(self.G):#Generate enough instantiation of GateWays  
            ll = i +1

            gw = GateWay(env,'GW%02d' %i,0.00001)
            G = 'G' + str(ll)
            GateWays[G] = gw


       

        MNet = MixNet(env,Mixes,GateWays)  #Generate an instantiation of the mix net
        random.seed(42)  
        

        Process = Message_Genartion_and_mix_net_processing(env,Mixes,capacity,Mix_Dict,MNet,self.N_target,self.d2,self.H_N,self.rate)

        env.process(Process.Prc())  #process the simulation

        env.run(until = self.run)  #Running time

        
        Latencies = MNet.LL
       
        Latencies_T = MNet.LT
        Distributions = np.matrix(MNet.EN)
        DT = np.transpose(Distributions)
        ENT = []

        for i in range(self.N_target):
            llll = DT[i,:].tolist()[0]
            ENT.append(Ent(llll))
        return Latencies, Latencies_T,ENT


    
    def truncate(self,alpha,List,Latency_List,Threshold):
        if alpha ==0:
            LIST = [List[0]]
            for i in range(1,len(List)):
                if Latency_List[List[i]]<Threshold:
                    LIST.append(List[i])
        elif alpha ==1:
            LIST = List
        else:
            a = int(alpha*len(List)+1)
            if a> len(List):
                a == len(List)
            LIST = List[0:a]
            for i in range(a,len(List)):
                if Latency_List[List[i]]<Threshold:
                    LIST.append(List[i])
        return LIST
  
    def PreProcessing(self,Iteration,Name,Common = False):
        import time
        self.Var =  [0.0,0.001,0.005,0.01,0.05,0.1]
        File_name = Name        
        import os         
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))          
        
        Delta = self.Var
        DATA = self.Circles_Creation(Iteration,Common)

        data = {}
        
        for k in range(len(DATA)):
            
            dict1 = {}
            dict1['Close'] = DATA['Iteration'+str(k)]['Close']
            for j in range(self.G):
                List = DATA['Iteration'+str(k)]['G'+str(j)][0]
                INDEX = DATA['Iteration'+str(k)]['G'+str(j)][1]
                
                dict2 = {}
                interval = (0.5)/5
                for i in range(6):
                    alpha = self.Var[i]
                    if i==0:
                        Threshold = 0.001
                    else:
                        Threshold = Delta[i]
                    Index = self.truncate(0.02,INDEX,List,Threshold)
                    t1 = time.time()
                    PDF1 = self.PDF(List,Index)
                    t2 =time.time()
                    PDF2 = self.PDF(List,Index,True,False)  
                    t3 =time.time()
                    PDF3 = self.PDF(List,Index,True,True) 
                    t4 = time.time()
                    self.t_u = (t2-t1)
                    self.t_p = (t3-t2)
                    self.t_l = (t4-t3)
                    
                    dict2[str(alpha)+'Uniform'] = [List,PDF1]
                    dict2[str(alpha)+'Fair'] = [List,PDF2]
                    dict2[str(alpha)+'LARMIX'] = [List,PDF3]
                    
                    PDF4 = self.PDF__(List,Index,0.3) 
                    PDF5 = self.PDF__(List,Index,0.9)    
                    dict2[str(alpha)+'LARMIX'+str(0.3)] = [List,PDF4]
                    dict2[str(alpha)+'LARMIX'+str(0.9)] = [List,PDF5]                    
                    
                dict1['G'+str(j+1)] = dict2
            for j in range(2*self.W):
                List = DATA['Iteration'+str(k)]['PM'+str(j+1)][0]
                INDEX = DATA['Iteration'+str(k)]['PM'+str(j+1)][1]
                interval = (max(List))/5
                
                dict2 = {}
                
                for i in range(6):
                    alpha = self.Var[i]
                    if i==0:
                        Threshold = 0.001
                    else:
                        Threshold = Delta[i]
                    Index = self.truncate(0.02,INDEX,List,Threshold)
                    PDF1 = self.PDF(List,Index)
                    PDF2 = self.PDF(List,Index,True,False)                
                    PDF3 = self.PDF(List,Index,True,True) 
                    
                    dict2[str(alpha)+'Uniform'] = [List,PDF1]
                    dict2[str(alpha)+'Fair'] = [List,PDF2]
                    dict2[str(alpha)+'LARMIX'] = [List,PDF3]
                    PDF4 = self.PDF__(List,Index,0.3) 
                    PDF5 = self.PDF__(List,Index,0.9)    
                    dict2[str(alpha)+'LARMIX'+str(0.3)] = [List,PDF4]
                    dict2[str(alpha)+'LARMIX'+str(0.9)] = [List,PDF5]                      
                dict1['PM'+str(j+1)] = dict2 
            #dict1['CNode'] = DATA['Iteration'+str(k)]['CNode']
                
                
                
            data['Iteration'+str(k)] = dict1
            
        import pickle   
        file_name = Name+'/data.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)
        return data
    
    
    
    
    def PreProcessing_(self,Iteration,Name,Common = False):
        import time
        #self.Var =  [0.0,0.001,0.005,0.01,0.05,0.1]
        self.Tau = [15/1000]
        self.ALPHA = [0.02,0.05,0.1,0.15,0.2,0.3]
        File_name = Name        
        import os         
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))          
        
        Delta = self.Var
        DATA = self.Circles_Creation(Iteration,Common)

        data = {}
        
        for k in range(len(DATA)):
            
            dict1 = {}
            dict1['Close'] = DATA['Iteration'+str(k)]['Close']
            for j in range(self.G):
                List = DATA['Iteration'+str(k)]['G'+str(j)][0]
                INDEX = DATA['Iteration'+str(k)]['G'+str(j)][1]
                
                dict2 = {}
                interval = (0.5)/5
                for i in range(6):
                    alpha = 15/1000
                    if i==0:
                        Threshold = 0.001
                    else:
                        Threshold = alpha
                    Index = self.truncate(self.ALPHA[i],INDEX,List,Threshold)
                    t1 = time.time()
                    PDF1 = self.PDF(List,Index)
                    t2 =time.time()
                    PDF2 = self.PDF(List,Index,True,False)  
                    t3 =time.time()
                    PDF3 = self.PDF(List,Index,True,True) 
                    t4 = time.time()
                    self.t_u = (t2-t1)
                    self.t_p = (t3-t2)
                    self.t_l = (t4-t3)
                    
                    dict2[str(self.ALPHA[i])+'Uniform'] = [List,PDF1]
                    dict2[str(self.ALPHA[i])+'Fair'] = [List,PDF2]
                    dict2[str(self.ALPHA[i])+'LARMIX'] = [List,PDF3]
                    
                    PDF4 = self.PDF__(List,Index,0.3) 
                    PDF5 = self.PDF__(List,Index,0.9)    
                    dict2[str(self.ALPHA[i])+'LARMIX'+str(0.3)] = [List,PDF4]
                    dict2[str(self.ALPHA[i])+'LARMIX'+str(0.9)] = [List,PDF5]                    
                    
                dict1['G'+str(j+1)] = dict2
            for j in range(2*self.W):
                List = DATA['Iteration'+str(k)]['PM'+str(j+1)][0]
                INDEX = DATA['Iteration'+str(k)]['PM'+str(j+1)][1]
                interval = (max(List))/5
                
                dict2 = {}
                
                for i in range(6):
                    alpha = 15/1000
                    if i==0:
                        Threshold = 0.001
                    else:
                        Threshold = alpha
                    Index = self.truncate(self.ALPHA[i],INDEX,List,Threshold)
                    PDF1 = self.PDF(List,Index)
                    PDF2 = self.PDF(List,Index,True,False)                
                    PDF3 = self.PDF(List,Index,True,True) 
                    
                    dict2[str(self.ALPHA[i])+'Uniform'] = [List,PDF1]
                    dict2[str(self.ALPHA[i])+'Fair'] = [List,PDF2]
                    dict2[str(self.ALPHA[i])+'LARMIX'] = [List,PDF3]
                    PDF4 = self.PDF__(List,Index,0.3) 
                    PDF5 = self.PDF__(List,Index,0.9)    
                    dict2[str(self.ALPHA[i])+'LARMIX'+str(0.3)] = [List,PDF4]
                    dict2[str(self.ALPHA[i])+'LARMIX'+str(0.9)] = [List,PDF5]                      
                dict1['PM'+str(j+1)] = dict2 
            #dict1['CNode'] = DATA['Iteration'+str(k)]['CNode']
                
                
                
            data['Iteration'+str(k)] = dict1
            
        import pickle   
        file_name = Name+'/data.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)
        return data
 
    
    def Entropy_Transformation(self,T):
        
        H = []
        for i in range(self.W):
            List = []
            for k in range(self.W):
                List.append(T[i,k])
            L =[]
            for item in List:
                if item!=0:
                    L.append(item)
            l = sum(L)
            for i in range(len(L)):
                L[i]=L[i]/l
            ent = 0
            for item in L:
                ent = ent - item*(np.log(item)/np.log(2))
            H.append(ent)
        if sum(H) == 0:
            return 0
        for j in range(len(H)):
            H[j] = H[j]/len(H)
        Entropy = sum(H) 
        
        return Entropy
    
    def make_T(self,G1,G2,G3):
        import numpy as np
        g1 = np.matrix(G1)
        g2 = np.matrix(G2)
        g3 = np.matrix(G3)
        return g1.dot(g2).dot(g3)
        
    def Analytic_Entropy(self,Data,Yes = False):
        import numpy as np
        Entropy_ = {}
        Load = {}
        for I in ['Uniform','LARMIX'+str(0.9),'Fair','LARMIX','LARMIX'+str(0.3)]:
            for J in range(6):
                tau = self.Var[J]
                Entropy_[str(tau)+I] = []
                Load[str(tau)+I] = []
        for i in range(len(Data)):
            data = Data['Iteration'+str(i)]

            for j in range(6):
                alpha = self.Var[j]
                #print(alpha)
                Gamma11 = []
                Gamma21 = []
                Gamma31 = []
                Gamma12 = []
                Gamma22 = []
                Gamma32 = []
                Gamma13 = []
                Gamma23 = []
                Gamma33 = []   
                
                Gamma14 = []
                Gamma15 = []
                Gamma24 = []
                Gamma25 = []
                Gamma34 = []
                Gamma35 = []
                for j in range(self.G):
                    Gamma11.append(data['G'+str(j+1)][str(alpha)+'Uniform'][1])
                    Gamma12.append(data['G'+str(j+1)][str(alpha)+'Fair'][1])                    
                    Gamma13.append(data['G'+str(j+1)][str(alpha)+'LARMIX'][1]) 

                    Gamma14.append(data['G'+str(j+1)][str(alpha)+'LARMIX'+str(0.3)][1]) 
                    Gamma15.append(data['G'+str(j+1)][str(alpha)+'LARMIX'+str(0.9)][1]) 
                    
                for j in range(self.W):
                    Gamma21.append(data['PM'+str(j+1)][str(alpha)+'Uniform'][1])
                    Gamma22.append(data['PM'+str(j+1)][str(alpha)+'Fair'][1])                    
                    Gamma23.append(data['PM'+str(j+1)][str(alpha)+'LARMIX'][1])                     


                    Gamma24.append(data['PM'+str(j+1)][str(alpha)+'LARMIX'+str(0.3)][1]) 
                    Gamma25.append(data['PM'+str(j+1)][str(alpha)+'LARMIX'+str(0.9)][1])                    
                for j in range(self.W,2*(self.W)):
                    Gamma31.append(data['PM'+str(j+1)][str(alpha)+'Uniform'][1])
                    Gamma32.append(data['PM'+str(j+1)][str(alpha)+'Fair'][1])                    
                    Gamma33.append(data['PM'+str(j+1)][str(alpha)+'LARMIX'][1])  
                    
                    Gamma34.append(data['PM'+str(j+1)][str(alpha)+'LARMIX'+str(0.3)][1]) 
                    Gamma35.append(data['PM'+str(j+1)][str(alpha)+'LARMIX'+str(0.9)][1]) 

                threshold = 1/self.W
                T_0 = self.make_T(Gamma11,Gamma21,Gamma31)
                LIST_LOAD = self.filter_matrix_entries(Gamma11, threshold)+self.filter_matrix_entries(Gamma21, threshold)+self.filter_matrix_entries(Gamma31, threshold)
                max_load = (np.sum(LIST_LOAD))/(len(LIST_LOAD))
                Load[str(alpha)+'Uniform'].append(max_load)
                E_0 = self.Entropy_Transformation(T_0)
                Entropy_[str(alpha)+'Uniform'].append(E_0)
                T_1 = self.make_T(Gamma12,Gamma22,Gamma32) 
                LIST_LOAD = self.filter_matrix_entries(Gamma12, threshold)+self.filter_matrix_entries(Gamma22, threshold)+self.filter_matrix_entries(Gamma32, threshold)
                max_load = (np.sum(LIST_LOAD))/(len(LIST_LOAD))   
                Load[str(alpha)+'Fair'].append(max_load)                   
                E_1 = self.Entropy_Transformation(T_1)  
                Entropy_[str(alpha)+'Fair'].append(E_1)                  
                T_2 = self.make_T(Gamma13,Gamma23,Gamma33)  
                LIST_LOAD = self.filter_matrix_entries(Gamma13, threshold)+self.filter_matrix_entries(Gamma23, threshold)+self.filter_matrix_entries(Gamma33, threshold)
                max_load = (np.sum(LIST_LOAD))/(len(LIST_LOAD))   
                Load[str(alpha)+'LARMIX'].append(max_load)                  
                E_2 = self.Entropy_Transformation(T_2)                   
                Entropy_[str(alpha)+'LARMIX'].append(E_2) 
           
                
                
                T_3 = self.make_T(Gamma14,Gamma24,Gamma34) 
                LIST_LOAD = self.filter_matrix_entries(Gamma14, threshold)+self.filter_matrix_entries(Gamma24, threshold)+self.filter_matrix_entries(Gamma34, threshold)
                max_load = (np.sum(LIST_LOAD))/(len(LIST_LOAD))   
                Load[str(alpha)+'LARMIX'+str(0.3)].append(max_load)                 
                E_3 = self.Entropy_Transformation(T_3)                   
                Entropy_[str(alpha)+'LARMIX'+str(0.3)].append(E_3)    
                
                
                T_4 = self.make_T(Gamma15,Gamma25,Gamma35) 
                LIST_LOAD = self.filter_matrix_entries(Gamma15, threshold)+self.filter_matrix_entries(Gamma25, threshold)+self.filter_matrix_entries(Gamma35, threshold)
                max_load = (np.sum(LIST_LOAD))/(len(LIST_LOAD))   
                Load[str(alpha)+'LARMIX'+str(0.9)].append(max_load)                  
                E_4 = self.Entropy_Transformation(T_4)                   
                Entropy_[str(alpha)+'LARMIX'+str(0.9)].append(E_4)                  
                
        LIST__ = []
        load_list = []
        for I in ['Uniform','Fair','LARMIX'+str(0.9),'LARMIX','LARMIX'+str(0.3)]:
            for J in range(6):   
                alpha = self.Var[J]
                LIST__.append(Entropy_[str(alpha)+I])
                load_list.append(Load[str(alpha)+I])
        Med_ = Med(LIST__,50)
        med_ = Med(load_list,50)
        
        if Yes:
            return [Med_[0:6],Med_[6:12],Med_[12:18],Med_[18:24],Med_[24:30]],[med_[0:6],med_[6:12],med_[12:18],med_[18:24],med_[24:30]]
        
        else:
            return [Med_[0:6],Med_[6:12],Med_[12:18],Med_[18:24],Med_[24:30]]
                
                
            
                    
                
                    
                    
                    
                    
                    
                    
                    
                    
                    
                
                
    
    def Latency_Med(self,List):
        import numpy as np
        a = np.transpose(np.matrix(List))
        return Med(a.tolist())
    

    
    def percentile_from_probabilities(self,Factors, Delays, percentile):
        import numpy as np
        T = np.array(np.transpose(np.matrix([Factors,Delays])).tolist())
        TT = T[T[:,1].argsort()]
        probabilities = TT[:,0].tolist()
        outcomes      = TT[:,1].tolist()
        cumulative_probabilities = np.cumsum(probabilities)
        
        R = np.abs(np.array(cumulative_probabilities)-percentile).tolist()
        Index = R.index(min(R))
    
        return outcomes[Index]
                    
    def Analytic_Latency(self,Data1, t = False):
        R = ['Uniform','LARMIX'+str(0.9),'Fair','LARMIX','LARMIX'+str(0.3)]
        output1 = []
        output2 = []
        for Routing in R:
            if t:
                o1,o2 = self.Analytic_Latency_(Data1,Routing)
            elif not t:
                o1,o2 = self.Analytic_Latency_normal(Data1,Routing)
                
            output1.append(o1)
            output2.append(o2)          


        return output1,output2

    def Analytic_Latency_Trade_Off(self,Data1,Name, t = False):
        R = [Name]
        output1 = []
        output2 = []
        for Routing in R:
            if t:
                o1,o2 = self.Analytic_Latency_(Data1,Routing)
            elif not t:
                o1,o2 = self.Analytic_Latency_normal(Data1,Routing)
                
            output1.append(o1)
            output2.append(o2)          


        return output1[0],output2[0]
            
    def Analytic_Entropy_Trade_Off(self,Data,Name):
        Entropy_ = {}
        for I in [Name]:
            for J in range(6):
                Entropy_[str(self.Var[J])+I] = []
        for i in range(len(Data)):
            data = Data['Iteration'+str(i)]

            for j in range(6):
                alpha = self.Var[j]
                Gamma11 = []
                Gamma21 = []
                Gamma31 = []

                for j in range(self.G):
                    Gamma11.append(data['G'+str(j+1)][str(alpha)+Name][1])

                    
                for j in range(self.W):
                    Gamma21.append(data['PM'+str(j+1)][str(alpha)+ Name][1])
                  
                for j in range(self.W,2*(self.W)):
                    Gamma31.append(data['PM'+str(j+1)][str(alpha)+Name][1])

                T_0 = self.make_T(Gamma11,Gamma21,Gamma31)
                E_0 = self.Entropy_Transformation(T_0)
                Entropy_[str(alpha)+Name].append(E_0)
                  
                
        LIST__ = []
        for I in [Name]:
            for J in range(6):     
                LIST__.append(Entropy_[str(self.Var[J])+I])
        Med_ = Med(LIST__,50)

        return Med_[0:6]
                
                
    def Analytic_Latency_(self,Data1,Routing):

        U__ = []


        for Iteration in range(len(Data1)):
            
            Data = Data1['Iteration'+str(Iteration)]
            U1 = []
            U2 = []
            U3 = []

            for II in range(6):
                alpha = self.Var[II]

                
                Y1 = []
                Y2 = []
                Y3 = []
                for i in range(self.G):
                    y1 = 0
                    for j in range(self.W):
                        x = Data['G'+str(i+1)][str(alpha)+Routing][0][j]
                        x = Data['G'+str(i+1)][str(alpha)+Routing][1][j]*x
                        y1 = y1+x
                    Y1.append(y1)
                        
                        
                        
                for j in range(self.W):
                    y2=0
                    for k in range(self.W,2*self.W):
                        x =  Data['PM'+str(j+1)][str(alpha)+Routing][0][k-self.W]
                        x = Data['PM'+str(j+1)][str(alpha)+Routing][1][k-self.W]*x
                        y2 = y2+x
                    Y2.append(y2)                        
                for k in range(self.W,2*self.W):
                    y3 = 0
                    for z in range(2*self.W,3*self.W): 
                        x = Data['PM'+str(k+1)][str(alpha)+Routing][0][z-2*self.W]
                        x = x*Data['PM'+str(k+1)][str(alpha)+Routing][1][z-2*self.W]
                        y3 = y3+x
                    Y3.append(y3)
                U1.append(Y1)                               
                U2.append(Y2)                                
                U3.append(Y3)                                
                           
                              
                        
                              
                                


            U__.append([U1,U2,U3])
        output1 = []
        for i in range(6):
            List1 = []
            
            for j in range(len(U__)):
                
                List1 = List1 + U__[j][0][i]
            output1.append(List1)
            
        output2 = []
        for i in range(6):
            List2 = []
          
            for j in range(len(U__)):             
                List2 = List2 + U__[j][1][i]
            output2.append(List2)                
                
        output3 = []
        for i in range(6):
            List3 = []
          
            for j in range(len(U__)):             
                List3 = List3 + U__[j][2][i]
            output3.append(List3)            
            #Output = self.Latency_Med(U__)   
            
        output = np.matrix(Med(output1,self.PP[0])) + np.matrix(Med(output2,self.PP[0])) +np.matrix(Med(output3,self.PP[0]))
            

        return  output.tolist()[0], (np.matrix(Med(output1,self.PP[1]))*3).tolist()[0]                        
                                

        
            

    def Analytic_Latency_normal(self,Data1,Routing):
        import numpy as np

        U__ = []


        for Iteration in range(len(Data1)):
            
            Data = Data1['Iteration'+str(Iteration)]
            U = []

            for II in range(6):
                alpha = self.Var[II]

                S = 0
        

                for i in range(self.G):
                    for j in range(self.W):
                        for k in range(self.W,2*self.W):
                            for z in range(2*self.W,3*self.W):
                                
                                x = Data['G'+str(i+1)][str(alpha)+Routing][0][j] + Data['PM'+str(j+1)][str(alpha)+Routing][0][k-self.W] +Data['PM'+str(k+1)][str(alpha)+Routing][0][z-2*self.W]
                                y = (1/self.W)*Data['G'+str(i+1)][str(alpha)+Routing][1][j]*Data['PM'+str(j+1)][str(alpha)+Routing][1][k-self.W]*Data['PM'+str(k+1)][str(alpha)+Routing][1][z-2*self.W]
                                S = S + x*y
                U.append(S)
            U__.append(U)
                        
                        
                        
                    
                              
                                



        output = np.transpose(np.matrix(U__)).tolist()

            

        return  Med(output,self.PP[0]), Med(output,self.PP[1])                        
                                

    def Analytic_Latency_Slow_Fast(self,FileName):
        Tau = self.Var
        Tau[0] = 0.01
        self.W = 32
        self.G = self.W
        
        Layer1 = []
        Layer2 = []
        Layer3 = []
        Policy1 = {}
        Policy2 = {}
        Policy3 = {}
        
        Gates,Mixes = self.MixNet_Creation() 
        
       
        
        
        
        
        for i in range(self.G):
            layer1 = []
            layer2 = []
            layer3 = []
            for j in range(self.W):
                layer1.append(Gates['G'+str(i+1)+'PM'+str(j+1)])
                layer2.append(Mixes['Layer1']['PM'+str(i+1)+'PM'+str(j+1+self.W)])
                layer3.append(Mixes['Layer2']['PM'+str(self.W+i+1)+'PM'+str(2*self.W+j+1)])
            Layer1.append(layer1)
            Layer2.append(layer2)
            Layer3.append(layer3)
                
        for tau in Tau:
            policy1 = []
            policy2 = []
            policy3 = []
                    
                
            for i in range(len(Layer1)):
                policy1.append(self.LARMIX(Layer1[i],tau))
                policy2.append(self.LARMIX(Layer2[i],tau))
                policy3.append(self.LARMIX(Layer3[i],tau))
        
            Policy1['tau'+str(tau)]= policy1
            Policy2['tau'+str(tau)]= policy2  
            Policy3['tau'+str(tau)]= policy3 
        t1 = []
        t2 = []
        import time
        
        Ave1 = []
        for tau in Tau:
            T1 = time.time()
            ave = 0
            for i in range(self.W):
                for j in range(self.W):
                    for k in range(self.W):
                        for z in range(self.W):
                            Latency_ = Layer1[i][j]+Layer2[j][k]+Layer3[k][z]
                            Pro_ = (1/self.W)*Policy1['tau'+str(tau)][i][j]*Policy2['tau'+str(tau)][j][k]*Policy3['tau'+str(tau)][k][z]
                            ave = ave + Latency_*Pro_
            Ave1.append(ave)
                            
            T2 = time.time()
            t1.append(T2-T1)
        

                            
        
        
        Ave2 = []
        for tau in Tau:
            ave1 = 0
            ave2 = 0
            ave3 = 0
            T3 = time.time()
            for i in range(self.W):
                for j in range(self.W):
                    ave1 = ave1 +  Layer1[i][j]*Policy1['tau'+str(tau)][i][j] 
            for j in range(self.W):
                for k in range(self.W):
                    ave2 = ave2 +  Layer2[j][k]*Policy2['tau'+str(tau)][j][k]                   
                    
            for k in range(self.W):
                for z in range(self.W):
                    ave3 = ave3 +  Layer1[k][z]*Policy3['tau'+str(tau)][k][z]   
            Ave2.append((ave1+ave2+ave3)/self.W)
                    

                            
            T4 = time.time()   
            t2.append(T4-T3)
        import os         
        if not os.path.exists(FileName):
            os.mkdir(os.path.join('', FileName))        
        from Plot import PLOT
        labels = Tau
        D = ['Estimated Latency','Exact Latency']
        f = ['Estimated Latency V.S. Exact Latency']
##################################Plots##################################################           
        Y = [Ave2,Ave1 ]
        Y_Label = 'Latency (sec)'
        X_Label = r'$\tau  $'
        Name = FileName + '/' +  'SlowFast_Latency.png'
        Name2 = FileName + '/' + 'SlowFast_time.png'
        PLT = PLOT(labels,Y,D,X_Label,Y_Label,Name)


        PLT.scatter_line(True,0.2)  
        import numpy as np
        t = np.matrix(t1)/np.matrix(t2)
        t = t.tolist()

        PLT2 = PLOT(labels,t,f,X_Label,'Time(sec)',Name2)


        PLT2.scatter_line(True,2000)  









    def Compete(self,Name_,Iteration):
        
        import numpy as np
        import time
        import json
        File_name = Name_        
        import os         
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))     
        import pickle
        Dictionaries = self.PreProcessing(Iteration,Name_)
        
        
        
       # with open(File_name+'\data.pkl', 'rb') as file:
            #Dictionaries = pickle.load(file)
        a = time.time()
        Latencyy, qqq = self.Analytic_Latency(Dictionaries,True) 
        b = time.time()
        Latency_A1,Latency_A2 = self.Analytic_Latency(Dictionaries) 
        c = time.time()
        Y = [Latencyy,Latency_A1]
       # print(Y)
        from Plot import PLOT
        labels = [0.0,0.001,0.005,0.01,0.05,0.1]
        D = ['Estimated Latency','Exact Latency']
        f = ['Estimated Latency V.S. Exact Latency']
##################################Plots##################################################           
       
        
        Y_Label = 'Latency (sec)'
        X_Label = r'$\tau  $'
        Name = Name_ + '/' +  'SlowFast_Latency.png'
        Name2 = Name_ + '/' + 'SlowFast_time.png'
        PLT = PLOT(labels,Y,D,X_Label,Y_Label,Name)


        PLT.scatter_line(True,0.2)       
        return (c-b)/(b-a)
        
    def FCP_Greedy(self,data,G_mean,Type):
        from FCP_ import FCP_Mix
        C = FCP_Mix(data,self.Adversary_Budget)
        if Type=='Random':
            C_nodes,FCP = C.C_random(G_mean)
        elif Type=='Close':
            C_nodes,FCP = C.Close_knit_nodes(G_mean,self.T_data)            
        elif Type=='Greedy':
            C_nodes,FCP = C.Greedy_For_Fairness(G_mean) 
        return [C_nodes, FCP]
    
    def FCP(self,Iteration,Name_,Common=False):
        import numpy as np
        IT = 'Iteration'
        Dictionaries = self.PreProcessing(Iteration,Name_,Common)
        Var = self.Var
        Names = ['Uniform','Fair','LARMIX0.3','LARMIX','LARMIX0.9']
        Names__ = ['LARMIX0.3','LARMIX','Fair','LARMIX0.9','Uniform']
        Methods = ['Random','Close','Greedy']
        output__ = {}
        for ii in range(Iteration):
            output__[IT+str(ii)] = {}
            Dict_FCP = {'output_Random_FCP':{},'output_Greedy_FCP':{},'output_Close_FCP':{}}
            Dict_CN = {'output_Random_CN':{},'output_Greedy_CN':{},'output_Close_CN':{}}
   
            
            for item in Names:
            
                for term in Var:
                    #print(term)
                    G_dist = []
                    for k in range(self.W):
                        #print(Dictionaries[IT+str(ii)]['G'+str(k+1)])

                        #for key in Dictionaries[IT+str(ii)]['G'+str(k+1)]:
                            #print(key)
                        G_dist.append(Dictionaries[IT+str(ii)]['G'+str(k+1)][str(term)+item][1])
                        
                    G_matrix = np.matrix(G_dist)
                    G_mean = np.mean(G_matrix,axis=0).tolist()[0]
                    Input = {}
                    for i in range(2*self.W):
                        Input['PM' + str(i+1)] = Dictionaries[IT+str(ii)]['PM'+str(i+1)][str(term)+item][1]
                    Dict_output = {}    
                    for method in Methods:
                        if not method == 'Close':
                            Dict_output['output'+method] = self.FCP_Greedy(Input,G_mean,method)
                            

                        else:
                            self.T_data = Dictionaries[IT+str(ii)]['Close']
                            Dict_output['output'+method] = self.FCP_Greedy(Input,G_mean,method)

                        
                        Dict_FCP['output_' + method +'_FCP'][str(term)+item ]= Dict_output['output'+method][1]
                        Dict_CN['output_' + method +'_CN'][str(term)+item ]= Dict_output['output'+method][0]


            output__[IT+str(ii)]['CN'] = [Dict_CN['output_Random_CN'],Dict_CN['output_Close_CN'],Dict_CN['output_Greedy_CN'] ]
            output__[IT+str(ii)]['FCP'] = [Dict_FCP['output_Random_FCP'],Dict_FCP['output_Close_FCP'],Dict_FCP['output_Greedy_FCP']]
            
        AVE_FCP = {}
        for m_name in Methods:
            AVE_FCP[m_name] = {}
        for counter in range(len(Methods)):
            for item in Names:
                for term in Var:
                    A = []
                    for j in range(Iteration):
                        
                        
                            
                        b = output__[IT+str(j)]['FCP'][counter][str(term)+item]

                        A.append(b)
                    A_matrix = np.matrix(A)
                    A_mean = np.mean(A,axis=0)
                    AVE_FCP[Methods[counter]][str(term)+item] = A_mean

        import numpy as np
        import json
        File_name = Name_        
        import os         
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))  

        Y = {} 
        for m_name in Methods:
            Y[m_name] = {}
        for m_name in Methods:
            for name in Names__:
                Y[m_name][name] = []
                for term in self.Var:
                    Y[m_name][name].append(AVE_FCP[m_name][str(term)+name])
                    
                    
                    


        from Plot import PLOT      

        X_L = r'$\alpha = 0.02$'
        Y_t = 'Throughput'
        Y_L = "Fraction of Corrupted Paths"
        M_ = ['Random','Single Location','Worst Case']
        DD_ = ['Uniform','LARMIX'+ r'$\tau=$' + str(0.9),'Proportional','LARMIX'+r'$\tau=$'+str(0.6),'LARMIX'+r'$\tau=$'+str(0.3)] 
        DD = ['Proportional','LARMIX'+r'$\tau=$'+str(0.6),'LARMIX'+r'$\tau=$'+str(0.9),'LARMIX'+r'$\tau=$'+str(0.3),'Uniform'] 

        DD = ['LARMIX'+r'$\tau=$'+str(0.3),'LARMIX'+r'$\tau=$'+str(0.6),'Proportional','LARMIX'+r'$\tau=$'+str(0.9),'Uniform'] 
               
        Description = []
        Alpha = [0.001,0.007,0.015,0.030,0.05,0.1]
        Y_R = []
        Y_C = []
        Y_G = []
        for item in Y['Random']:
            Y_R.append(Y['Random'][item])
            Y_C.append(Y['Close'][item])
            Y_G.append(Y['Greedy'][item])            
        FCP_Random = File_name + '/'+'FCP_Random.png'
        FCP_Close = File_name + '/'+ 'FCP_Close.png'
        FCP_Greedy = File_name + '/' + 'FCP_Greedy.png' 
               
          
        PLT_Random = PLOT(Alpha,Y_R,DD,X_L,Y_L,FCP_Random)
        PLT_Random.scatter_line(True,0.1)


        PLT_Close = PLOT(Alpha,Y_C,DD,X_L,Y_L,FCP_Close)
        PLT_Close.scatter_line(True,0.1)


        PLT_Greedy = PLOT(Alpha,Y_G,DD,X_L,Y_L,FCP_Greedy)
        PLT_Greedy.scatter_line(True,0.5)               
                
        self.Simulation_FCP = output__
        FCP_Dicts = {'FCP':Y,'FCP_Sim':output__}
        #return AVE_FCP
        
        import json
        with open(File_name+'/FCP_Data.json','w') as file:
            json.dump(FCP_Dicts,file)




##########################Simulations#######################################################         
        Latency_alpha_Uniform = []
        Latency_alpha_Uniform_T = []    
        Entropy_alpha_Uniform = []
        Latency_alpha_Fair = []
        Latency_alpha_Fair_T = []    
        Entropy_alpha_Fair = []
        Latency_alpha_LARMIX = []
        Latency_alpha_LARMIX_T = []    
        Entropy_alpha_LARMIX = []
        corrupted_Mix = {}

        for k in range(self.N):
            corrupted_Mix['PM'+str(k+1)] = False


###########################################Larmix0.6 + Random ##############################   
        for j in range(6):
            alpha = self.Var[j]       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            

            for i in range(len(Dictionaries)):
                corrupted_Mix = output__['Iteration'+str(i)]['CN'][0][str(alpha)+'LARMIX0.9']
                            

                Mix_Dict = {}
                for I in range(self.G):
                    Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)]['G'+str(I+1)][str(alpha)+'LARMIX0.9']
                for J in range(2*self.W):
                    Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)]['PM'+str(J+1)][str(alpha)+'LARMIX0.9']



                    
                Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Mix_Dict)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_alpha_Uniform.append(End_to_End_Latancy_Vector)
            Latency_alpha_Uniform_T.append(End_to_End_Latancy_Vector_T)
            Entropy_alpha_Uniform.append(Message_Entropy_Vector)


            
            
            
            
            
            
            
            
            
            
            
            
            
            
###########################################Close + Larmix0.6 ##############################   
        for j in range(6):
            alpha = self.Var[j]       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            

            for i in range(len(Dictionaries)):
                corrupted_Mix = output__['Iteration'+str(i)]['CN'][1][str(alpha)+'LARMIX0.9']                            

                Mix_Dict = {}
                for I in range(self.G):
                    Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)]['G'+str(I+1)][str(alpha)+'LARMIX0.9']
                for J in range(2*self.W):
                    Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)]['PM'+str(J+1)][str(alpha)+'LARMIX0.9']


               
                Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Mix_Dict)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_alpha_Fair.append(End_to_End_Latancy_Vector)
            Latency_alpha_Fair_T.append(End_to_End_Latancy_Vector_T)
            Entropy_alpha_Fair.append(Message_Entropy_Vector)
            
            
            
            
###########################################Greedy + LARMIX0.6 ##############################   
        for j in range(6):
            alpha = self.Var[j]      
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            

            for i in range(len(Dictionaries)):
                corrupted_Mix = output__['Iteration'+str(i)]['CN'][2][str(alpha)+'LARMIX0.9']                            

                Mix_Dict = {}
                for I in range(self.G):
                    Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)]['G'+str(I+1)][str(alpha)+'LARMIX0.9']
                for J in range(2*self.W):
                    Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)]['PM'+str(J+1)][str(alpha)+'LARMIX0.9']



                Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Mix_Dict)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_alpha_LARMIX.append(End_to_End_Latancy_Vector)
            Latency_alpha_LARMIX_T.append(End_to_End_Latancy_Vector_T)
            Entropy_alpha_LARMIX.append(Message_Entropy_Vector)
            
            
            
            
            
            
            
            
            
##################################################################################
##################################################################################
            

        labels = [0.001,0.007,0.015,0.030,0.05,0.1]
        '''
        for Tau in np.arange(0,1.01,0.2):            
            T = round(100*Tau)/100
            labels.append(T)       
        for i in range(len(labels)):
          
            labels[i] = int(labels[i]*100)/100
            '''
###################################################################################            
#################################Saving the data###################################     
        df = {'Alpha':labels,
            'Latency_Uniform' : Latency_alpha_Uniform,
            'Entropy_Uniform' : Entropy_alpha_Uniform,     
            'Latency_Fair' : Latency_alpha_Fair,
            'Entropy_Fair' : Entropy_alpha_Fair, 
            'Latency_LARMIX' : Latency_alpha_LARMIX,
            'Entropy_LARMIX' : Entropy_alpha_LARMIX
                              }
        import json

        dics = json.dumps(df)
        with open(File_name + '/'+ 'FCP' +'Sim.json','w') as df_sim:
            json.dump(dics,df_sim)        
        
        
        
        
        D = ['Random','Single Location','Worst Case']
##################################Plots##################################################           
        Y = [Latency_alpha_Uniform ,Latency_alpha_Fair, Latency_alpha_LARMIX ]
        Y_Label = 'Latency (sec)'
        X_Label = 'Radius of cells' +r'$\alpha = 0.02 $'
        Name = File_name + '/' + 'FCP_Latency.png'
        from Plot import PLOT
        PLT = PLOT(labels,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(1,True)
        
        
        Y = [Entropy_alpha_Uniform ,Entropy_alpha_Fair, Entropy_alpha_LARMIX ]
        Y_Label = 'Entropy (bits)'
        Name = File_name + '/' +'FCP_Entropy.png'
        PLT = PLOT(labels,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(15,True)



#################################################################################################






    def FCP_Budget(self,Iteration,Name_,Budget,Common=False):
        import numpy as np
        self.Adversary_Budget = Budget
        IT = 'Iteration'
        Dictionaries = self.PreProcessing(Iteration,Name_,Common)
        Var = [0.01]
        Names = ['Uniform','Fair','LARMIX0.3','LARMIX','LARMIX0.9']
        Names__ = ['LARMIX0.3','LARMIX','Fair','LARMIX0.9','Uniform']
        Methods = ['Greedy']
        output__ = {}
        for ii in range(Iteration):
            output__[IT+str(ii)] = {}
            Dict_FCP = {'output_Greedy_FCP':{}}
            Dict_CN = {'output_Greedy_CN':{}}
   
            
            for item in Names:
            
                for term in Var:
                    #print(term)
                    G_dist = []
                    for k in range(self.W):
                        #print(Dictionaries[IT+str(ii)]['G'+str(k+1)])

                        #for key in Dictionaries[IT+str(ii)]['G'+str(k+1)]:
                            #print(key)
                        G_dist.append(Dictionaries[IT+str(ii)]['G'+str(k+1)][str(term)+item][1])
                        
                    G_matrix = np.matrix(G_dist)
                    G_mean = np.mean(G_matrix,axis=0).tolist()[0]
                    Input = {}
                    for i in range(2*self.W):
                        Input['PM' + str(i+1)] = Dictionaries[IT+str(ii)]['PM'+str(i+1)][str(term)+item][1]
                    Dict_output = {}    
                    for method in Methods:
                        if not method == 'Close':
                            Dict_output['output'+method] = self.FCP_Greedy(Input,G_mean,method)
                            

                        else:
                            self.T_data = Dictionaries[IT+str(ii)]['Close']
                            Dict_output['output'+method] = self.FCP_Greedy(Input,G_mean,method)

                        
                        Dict_FCP['output_' + method +'_FCP'][str(term)+item ]= Dict_output['output'+method][1]
                        Dict_CN['output_' + method +'_CN'][str(term)+item ]= Dict_output['output'+method][0]


            output__[IT+str(ii)]['CN'] = [Dict_CN['output_Greedy_CN'] ]
            output__[IT+str(ii)]['FCP'] = [Dict_FCP['output_Greedy_FCP']]
            
        AVE_FCP = {}
        for m_name in Methods:
            AVE_FCP[m_name] = {}
        for counter in range(len(Methods)):
            for item in Names:
                for term in Var:
                    A = []
                    for j in range(Iteration):
                        
                        
                            
                        b = output__[IT+str(j)]['FCP'][counter][str(term)+item]

                        A.append(b)
                    A_matrix = np.matrix(A)
                    A_mean = np.mean(A,axis=0)
                    AVE_FCP[Methods[counter]][str(term)+item] = A_mean

        import numpy as np
        import json
        File_name = Name_        
        import os         
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))  

        Y = {} 
        for m_name in Methods:
            Y[m_name] = {}
        for m_name in Methods:
            for name in Names__:
                Y[m_name][name] = []
                for term in [0.01]:
                    Y[m_name][name].append(AVE_FCP[m_name][str(term)+name])
                    
                    
                    

        FCP_Dicts = {'FCP':Y,'FCP_Sim':output__}
        #return AVE_FCP
                     
                
        self.Simulation_FCP = output__
        FCP_Dicts = {'FCP':Y,'FCP_Sim':output__}
        #return AVE_FCP
        
        import json
        with open(File_name+'/FCP_Data.json','w') as file:
            json.dump(FCP_Dicts,file)



    def E2E(self,e2e,Iteration,Item,Name_,Common = False):
        #print(Item)
        Radius = [0.001,0.005,0.015,0.03,0.05]       
        import numpy as np
        import json
        File_name = Name_        
        import os         
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))     
        import pickle
        Dictionaries = self.PreProcessing(Iteration,Name_,Common)
        
        
        #print('ok')
       # with open(File_name+'\data.pkl', 'rb') as file:
            #Dictionaries = pickle.load(file)
        
        Entropy_A = self.Analytic_Entropy_Trade_Off(Dictionaries,Item)
        Latency_A1,Latency_A2 = self.Analytic_Latency_Trade_Off(Dictionaries,Item)
        Mix_delays = []
        for item in Latency_A1:
            Mix_delays.append((e2e - item)/3)
        
#################################################################################         
        Latency_alpha_ = []
        Latency_alpha_T = []    
        Entropy_alpha_ = []

        corrupted_Mix = {}

        for k in range(self.N):
            corrupted_Mix['PM'+str(k+1)] = False


###########################################Uniform ##############################   
        for j in range(6):
            self.d1 = Mix_delays[j]
            alpha = self.Var[j]       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            

            for i in range(len(Dictionaries)):
                            

                Mix_Dict = {}
                for I in range(self.G):
                    Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)]['G'+str(I+1)][str(alpha)+Item]
                for J in range(2*self.W):
                    Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)]['PM'+str(J+1)][str(alpha)+Item]



                    
                Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Mix_Dict)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_alpha_.append(End_to_End_Latancy_Vector)
            Latency_alpha_T.append(End_to_End_Latancy_Vector_T)
            Entropy_alpha_.append(Message_Entropy_Vector)
        
        Sim_L_mean = []
        Sim_E_mean = []
        import numpy as np
        for i in range(len(Latency_alpha_)):
            Sim_L_mean.append(np.mean(Latency_alpha_[i]))
            Sim_E_mean.append(np.mean(Entropy_alpha_[i]))
            
        #print(Sim_E_mean,Sim_L_mean,Entropy_A,Latency_A1,Mix_delays)
        
        df = {'Sim_E_mean':Sim_E_mean,
              'Sim_L_mean':Sim_L_mean,
              'Entropy_A':Entropy_A,
              'Latency_A1':Latency_A1,
              'Mix_delays':Mix_delays}
        
        import json
        with open(File_name+'/Data_Tradeoffs'+Item+'.json','w') as file:
            json.dump(df,file)




    def Load_Analysis(self,Name_,Iteration,Common = False):
        import numpy as np
        import json
        File_name = Name_        
        self.Var = [0.001,0.007,0.015,0.03,0.05,0.1]
        import os         
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))
        self.Var = [0.001,0.007,0.015,0.03,0.05,0.1]
        import pickle
        import time
        self.Var = [0.001,0.007,0.015,0.03,0.05,0.1]
        t1 = time.time()
        Dictionaries = self.PreProcessing(Iteration,Name_,Common)
        t2=time.time()
        self.Var = [0.001,0.007,0.015,0.03,0.05,0.1]

        
        
        
       # with open(File_name+'\data.pkl', 'rb') as file:
            #Dictionaries = pickle.load(file)
        self.Var = [0.001,0.007,0.015,0.03,0.05,0.1]
        Entropy_A,Loads = self.Analytic_Entropy(Dictionaries,True)
        self.Var = [0.001,0.007,0.015,0.03,0.05,0.1]
        from Plot import PLOT      

        X_L = r'$\alpha = 0.02$'

        Y_L = "Max Loads"
        DD = ['Uniform','LARMIX'+ r'$\tau=$' + str(0.9),'Proportional','LARMIX'+r'$\tau=$'+str(0.6),'LARMIX'+r'$\tau=$'+str(0.3)]       
 
        D = ['Uniform','Proportional','LARMix']
        Alpha = [0.001,0.007,0.015,0.03,0.05,0.1]

            
        Name_Load = File_name + '/' + str(self.PP[0])+'Load.png'
           
        PLT_E = PLOT(Alpha,Loads,DD,X_L,Y_L,Name_Load)
        PLT_E.scatter_line(True,0.3)


     
        import pickle        
        df = {'Alpha':Alpha,
            'Loads':Loads                          
                              
                              } 
        with open(File_name + '/'+'Analytical_Loads.pkl', 'wb') as file:
            # Serialize and save your data to the file
            pickle.dump(df, file)          
#################################################################################         
#################################################################################
#################################################################################


    def Time_Analysis(self,Name_,Iteration,Common = False):

        import time
        t1 = time.time()
        Dictionaries = self.PreProcessing(Iteration,Name_,Common)
        t2=time.time()
        
        return (t2-t1)/(len(self.Var)*5*Iteration)





    def EL_Analysis(self,Name_,Iteration,Common = False):
        import numpy as np
        import json
        File_name = Name_        
        import os         
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))     
        import pickle
        import time
        t1 = time.time()
        Dictionaries = self.PreProcessing(Iteration,Name_,Common)
        t2=time.time()
        #print(t2-t1)

        
        
        
       # with open(File_name+'\data.pkl', 'rb') as file:
            #Dictionaries = pickle.load(file)
        
        Entropy_A = self.Analytic_Entropy(Dictionaries)
        Latency_A1,Latency_A2 = self.Analytic_Latency(Dictionaries)
        Frac = []
        for I in range(len(Entropy_A)):
            Latency100 = np.matrix(Latency_A1[I])
            Entropy100 = np.matrix(Entropy_A[I])
            f1 = Entropy100/Latency100
            Frac.append(f1.tolist()[0])

        from Plot import PLOT      

        X_L = r'$\alpha = 0.02$'
        Y_t = 'Throughput'
        Y_E = "Entropy (bits)"
        Y_L = 'Latency (sec)'
        DD = ['Uniform','LARMIX'+ r'$\tau=$' + str(0.9),'Proportional','LARMIX'+r'$\tau=$'+str(0.6),'LARMIX'+r'$\tau=$'+str(0.3)]       
 
        D = ['Uniform','Proportional','LARMix']
        Alpha = [0.0,0.001,0.005,0.01,0.05,0.1]

            
        Name_Entropy = File_name + '/' + str(self.PP[0])+'Entropy.png'
        Name_Latency = File_name + '/'+ str(self.PP[0]) + 'Latency.png'
        Name_Latency_ = File_name + '/'+ str(self.PP[1]) + 'Latency.png'
        Name_t = File_name + '/'+ 'Throughput.png'            
        PLT_E = PLOT(Alpha,Entropy_A,DD,X_L,Y_E,Name_Entropy)
        PLT_E.scatter_line(True,7.5)


        PLT_t = PLOT(Alpha,Frac,DD,X_L,Y_t,Name_t)
        PLT_t.scatter_line(True,600)
        
        
        
        PLT_L1 = PLOT(Alpha,Latency_A1,DD,X_L,Y_L,Name_Latency)

        PLT_L1.scatter_line(True,0.15)
        PLT_L1 = PLOT(Alpha,Latency_A2,DD,X_L,Y_L,Name_Latency_)

        PLT_L1.scatter_line(True,0.4)        
        import pickle        
        df = {'Alpha':Alpha,
            'Latency':Latency_A1,'Entropy':Entropy_A,'Frac':Frac                            
                              
                              } 
        with open(File_name + '/'+'Analytical.pkl', 'wb') as file:
            # Serialize and save your data to the file
            pickle.dump(df, file)  
            
            
            
            
#################################################################################         
        #Latency_alpha_Uniform = []
        #Latency_alpha_Uniform_T = []    
        #Entropy_alpha_Uniform = []
        #Latency_alpha_Fair = []
        '''
        Latency_alpha_Fair_T = []    
        Entropy_alpha_Fair = []
        Latency_alpha_LARMIX = []
        Latency_alpha_LARMIX_T = []    
        Entropy_alpha_LARMIX = []
        corrupted_Mix = {}

        for k in range(self.N):
            corrupted_Mix['PM'+str(k+1)] = False


###########################################Uniform ##############################   
        for j in range(6):
            alpha =self.Var[j]       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            

            for i in range(len(Dictionaries)):
                            

                Mix_Dict = {}
                for I in range(self.G):
                    Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)]['G'+str(I+1)][str(alpha)+'Uniform']
                for J in range(2*self.W):
                    Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)]['PM'+str(J+1)][str(alpha)+'Uniform']



                    
                Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Mix_Dict)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_alpha_Uniform.append(End_to_End_Latancy_Vector)
            Latency_alpha_Uniform_T.append(End_to_End_Latancy_Vector_T)
            Entropy_alpha_Uniform.append(Message_Entropy_Vector)


            
###########################################Fair ##############################   
        for j in range(6):
            alpha = self.Var[j]       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            

            for i in range(len(Dictionaries)):
                            

                Mix_Dict = {}
                for I in range(self.G):
                    Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)]['G'+str(I+1)][str(alpha)+'Fair']
                for J in range(2*self.W):
                    Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)]['PM'+str(J+1)][str(alpha)+'Fair']


               
                Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Mix_Dict)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_alpha_Fair.append(End_to_End_Latancy_Vector)
            Latency_alpha_Fair_T.append(End_to_End_Latancy_Vector_T)
            Entropy_alpha_Fair.append(Message_Entropy_Vector)
            
            
            
            
###########################################LARMIX ##############################   
        for j in range(6):
            alpha = self.Var[j]       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            

            for i in range(len(Dictionaries)):
                            

                Mix_Dict = {}
                for I in range(self.G):
                    Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)]['G'+str(I+1)][str(alpha)+'LARMIX']
                for J in range(2*self.W):
                    Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)]['PM'+str(J+1)][str(alpha)+'LARMIX']



                Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Mix_Dict)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT       
            Latency_alpha_LARMIX.append(End_to_End_Latancy_Vector)
            Latency_alpha_LARMIX_T.append(End_to_End_Latancy_Vector_T)
            Entropy_alpha_LARMIX.append(Message_Entropy_Vector)
            
            
            
            
            
            
            
            
            
##################################################################################
##################################################################################
            

        labels = self.Var
        
        for Tau in np.arange(0,1.01,0.2):            
            T = round(100*Tau)/100
            labels.append(T)       
        for i in range(len(labels)):
          
            labels[i] = int(labels[i]*100)/100
            
###################################################################################            
#################################Saving the data###################################     
        df = {'Alpha':labels,
            'Latency_Uniform' : Latency_alpha_Uniform,
            'Entropy_Uniform' : Entropy_alpha_Uniform,     
            'Latency_Fair' : Latency_alpha_Fair,
            'Entropy_Fair' : Entropy_alpha_Fair, 
            'Latency_LARMIX' : Latency_alpha_LARMIX,
            'Entropy_LARMIX' : Entropy_alpha_LARMIX
                              }
        import json

        dics = json.dumps(df)
        with open(File_name + '/'+ str(self.PP[0]) +'Sim.json','w') as df_sim:
            json.dump(dics,df_sim)        
        
        
        
        
    
##################################Plots##################################################           
        Y = [Latency_alpha_Uniform ,Latency_alpha_Fair, Latency_alpha_LARMIX ]
        Y_Label = 'Latency (sec)'
        X_Label = 'Radius of cells' +r'$\alpha = 0.02 $'
        Name = File_name + '/' + str(self.PP[0])+ 'Sim_Latency.png'
        from Plot import PLOT
        PLT = PLOT(labels,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(10,True)
        
        
        Y = [Entropy_alpha_Uniform ,Entropy_alpha_Fair, Entropy_alpha_LARMIX ]
        Y_Label = 'Entropy (bits)'
        Name = File_name + '/' + str(self.PP[0])+'Sim_Entropy.png'
        PLT = PLOT(labels,Y,D,X_Label,Y_Label,Name)
        PLT.Box_Plot(1,True)
        '''



#################################################################################################







    def EL_Analysis_alpha(self,Name_,Iteration,Common = False):
        self.Var = [0.02,0.05,0.1,0.15,0.2,0.3]
        import numpy as np
        import json
        File_name = Name_        
        import os         
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))     
        import pickle
        import time
        t1 = time.time()
        Dictionaries = self.PreProcessing_(Iteration,Name_,Common)
        self.Var = [0.02,0.05,0.1,0.15,0.2,0.3]        
        t2=time.time()
        #print(t2-t1)

        
        
        
       # with open(File_name+'\data.pkl', 'rb') as file:
            #Dictionaries = pickle.load(file)
        self.Var = [0.02,0.05,0.1,0.15,0.2,0.3]        
        Entropy_A = self.Analytic_Entropy(Dictionaries)
        self.Var = [0.02,0.05,0.1,0.15,0.2,0.3]        
        Latency_A1,Latency_A2 = self.Analytic_Latency(Dictionaries)
        self.Var = [0.02,0.05,0.1,0.15,0.2,0.3]        
        Frac = []
        for I in range(len(Entropy_A)):
            Latency100 = np.matrix(Latency_A1[I])
            Entropy100 = np.matrix(Entropy_A[I])
            f1 = Entropy100/Latency100
            Frac.append(f1.tolist()[0])

        from Plot import PLOT      

        X_L = r'$\alpha = 0.02$'
        Y_t = 'Throughput'
        Y_E = "Entropy (bits)"
        Y_L = 'Latency (sec)'
        DD = ['Uniform','LARMIX'+ r'$\tau=$' + str(0.9),'Proportional','LARMIX'+r'$\tau=$'+str(0.6),'LARMIX'+r'$\tau=$'+str(0.3)]       
 
        D = ['Uniform','Proportional','LARMix']
        Alpha = [0.02,0.05,0.1,0.15,0.2,0.3]  

            
        Name_Entropy = File_name + '/' + str(self.PP[0])+'Entropy.png'
        Name_Latency = File_name + '/'+ str(self.PP[0]) + 'Latency.png'
        Name_Latency_ = File_name + '/'+ str(self.PP[1]) + 'Latency.png'
        Name_t = File_name + '/'+ 'Throughput.png'            
        PLT_E = PLOT(Alpha,Entropy_A,DD,X_L,Y_E,Name_Entropy)
        PLT_E.scatter_line(True,7.5)


        PLT_t = PLOT(Alpha,Frac,DD,X_L,Y_t,Name_t)
        PLT_t.scatter_line(True,600)
        
        
        
        PLT_L1 = PLOT(Alpha,Latency_A1,DD,X_L,Y_L,Name_Latency)

        PLT_L1.scatter_line(True,0.15)
        PLT_L1 = PLOT(Alpha,Latency_A2,DD,X_L,Y_L,Name_Latency_)

        PLT_L1.scatter_line(True,0.4)        
        import pickle        
        df = {'Alpha':Alpha,
            'Latency':Latency_A1,'Entropy':Entropy_A,'Frac':Frac                            
                              
                              } 
        with open(File_name + '/'+'Analytical.pkl', 'wb') as file:
            # Serialize and save your data to the file
            pickle.dump(df, file)          





















     
    '''        
           
                        
 #Exicution###################################################################
data = 'RIPE'
W = 60

N = 3*W
num_gateways = W
Goal = 'Circle'+str(W)
delay1 = 0.03
delay2 = 0.0001/5
Capacity = 10000000000000000000000000000000000000000000000000000000000000000
H_N = round(N/3)
rate = 100
num_targets = 200
Iterations = 20
run = 0.4
Percentile = [50,95]
Sim = CircularMixNet(num_targets,Iterations,Capacity,run,delay1,delay2,H_N,N,rate,num_gateways,Percentile,Goal) 

'''

#Sim.E2E(0.200,2,'LARMIX'+str(0.9),'Test_tredesoff',True)




#Sim.Analytic_Latency_Slow_Fast('Test2')



#GateWays,Layers = Sim.MixNet_Creation()
#GateWays,Layers = Sim.Distance_MG(MixNet1,GateWays1)
'''
import numpy as np
import json
Name_ = 'test00'
File_name = Name_        
import os         
if not os.path.exists(File_name):
    os.mkdir(os.path.join('', File_name))     
import pickle
Sim.PreProcessing(1,Name_)

with open(File_name+'\data.pkl', 'rb') as file:
    Dictionaries = pickle.load(file)

Latency_A1,Latency_A2 = Sim.Analytic_Latency(Dictionaries)
'''
#data = Sim.PreProcessing(1,'Test0')
#print(Sim.Analytic_Latency(data))
#
#print(Sim.Analytic_Entropy(data))
#Sim.Analytic_Latency(data)

#Sim.EL_Analysis('C5',5)


#a = Sim.Compete('C1',2)





#D = Sim.PreProcessing(1,'Test_FCP')



#Fraction_Of_Currpted_Paths
'''

FCP = Sim.FCP(5,'FCP_Simulation_and_Analytic',True)


print(FCP)


'''











