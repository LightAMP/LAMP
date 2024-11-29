# -*- coding: utf-8 -*-
"""
Regional Mixnet: The Regional Mixnet approach aims to reduce high-latency links in mixnets by dividing the network into
 distinct geographical regions. This approach segments the world into regions, ensuring that communication occurs primarily
 within low-latency regional boundaries. This file provides the implementation, simulation, and evaluation of the Regional 
 Mixnet approach.
"""
from mpl_toolkits.basemap import Basemap
from Plot import PLOT      
from math import exp
from scipy import constants
import time
import statistics
# Import library for making the simulation, making random choices,
#creating exponential delays, and defining matrixes.
from scipy.stats import expon
import simpy
import random
#from Datasets_ import Dataset
import numpy  as np
import pickle
import json
from Message_ import message
import math
from GateWay import GateWay
from Mix_Node_ import Mix
from FCP_ import FCP_Mix
from NYM import MixNet
from itertools import combinations
from math import radians, sin, cos, sqrt, atan2
from math import exp
from scipy import constants
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap 
# Import library for making the simulation, making random choices,
#creating exponential delays, and defining matrixes.
from scipy.stats import expon
import simpy
import random
import numpy  as np
import pickle
#from Datasets_ import Dataset
from Message_ import message

from GateWay import GateWay
from Mix_Node_ import Mix

from NYM import MixNet

from Message_Genartion_and_mix_net_processing_ import Message_Genartion_and_mix_net_processing
import plotly.express as px
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd



    
    
def convert_coordinates(latitudes, longitudes):
    



    data = {'lat': latitudes, 'lon': longitudes}
    df = pd.DataFrame(data)
    return df
    
    
    
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

    N = len(List)

    List_ = []

    for i in range(N):

        List_.append( np.percentile(List[i], Per))
        
    return List_





class Regional_MixNet(object):
    
    def __init__(self,Targets,Iteration,Capacity,run,delay1,delay2,H_N,N,rate,num_gateways,Percentile,Goal):
        self.Iterations = Iteration
        self.TYPE = 'RIPE'
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
        self.data_type = 'RIPE'
        self.PP = Percentile
        self.Var =  [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        self.GlobalData = {}
        self.colors = ['royalblue','red','green','fuchsia','cyan','indigo','teal','lime','blue','black','orange','violet','lightblue']
        self.Adversary_Budget = 30          

        self.Data_Clustering__ = self.Interpolated_NYM()  

        self.regions = ['1','2']
        self.close_data = {}
        for name in self.regions:
            self.close_data['Region'+name] = {}
            self.close_data['Global'+'Region'+name] = {}         
    def Latency_reader(self,a):
        b = ''
        for char in a:            
            if char=='m':
                break
            b = b +char
            if char =='u' or char=='n':
                b = '1'
                break
        return float(b)
    
  
    
    
    
    
    
    def Latency_reader_(self,item,Type):
        
        if Type =='NYM':
            
            output = self.Latency_reader(item)
            
        elif Type=='RIPE':
            delay_distance = int(item)
            if delay_distance == 0:
                delay_distance =1
            output = delay_distance
        
        return output
            
        
        
    def Latency_Matrix(self,data=False):
        

        
        with open('D:/Approach3/117_nodes_latency_December_2023_cleaned_up_9_no_intersection_1.json') as json_file: 
        
            data0 = json.load(json_file) # Your list of dictionaries
           
        A = np.zeros((len(data0),len(data0)))
        for i in range(len(data0)):
        
            for j in range(len(data0)):
                I_key = data0[j]['i_key']
                if not  i ==j:
                    
                    A[i,j] = self.Latency_reader_(data0[i]['latency_measurements'][str(I_key)],self.TYPE)/2000
                else:
                    A[i,j] = 0
        return A
             
             
    def Clustering(self,K_cluster = 0):

        latency_matrix = self.Latency_Matrix()
        
      
        
        with open('D:/Approach3/117_nodes_latency_December_2023_cleaned_up_9_no_intersection_1.json') as json_file: 
        
            data0 = json.load(json_file) # Your list of dictionaries
    
        # Generate a random 560x560 latency matrix
        np.random.seed(42)
        
        
        # Specify a range of cluster values
        k_values = range(2, 11)  # Choose a range of cluster values
        silhouette_scores = []
        
        # Perform KMeans clustering for each k value and calculate silhouette score
        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_assignments = kmeans.fit_predict(latency_matrix)
            silhouette_avg = silhouette_score(latency_matrix, cluster_assignments)
            silhouette_scores.append(silhouette_avg)
        
        # Plot the silhouette scores for different values of k

     

        X_L = 'Number of Clusters (k)'
        Y_t = 'Entropy/Latency'
        Y_E = "Entropy (bits)"
        Y_L = 'Silhouette Score'

        D = ['Silhouette Score for Optimal k']

            
        Name_K =  str(self.PP[0])+'Kclusters.png'
          
        PLT_K = PLOT(k_values,[silhouette_scores],D,X_L,Y_L,Name_K)
        PLT_K.scatter_line(True,0.6)
        
        # Choose the optimal number of clusters based on the elbow or another criterion
        if K_cluster == 0:
            
            optimal_k = k_values[np.argmax(silhouette_scores)]
        else:
            optimal_k = K_cluster
            
        
        # Perform KMeans clustering with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_assignments = kmeans.fit_predict(latency_matrix)

        #CENTERS = {'europe': 43, 'asia': 233, 'north_america': 4, 'south_america': 202, 'africa': 202, 'australia': 204}
        CENTERS = {'europe': 107, 'asia': 44, 'north_america': 21, 'south_america': 93, 'africa': 93, 'australia': 94}
        CENTERS_PRIORITY = ['europe','north_america','asia', 'south_america', 'africa', 'australia']
        initial_centers = latency_matrix
        rows_to_select = []
        for counter in range(optimal_k):
            rows_to_select.append(CENTERS[CENTERS_PRIORITY[counter]])
        initial_centers = latency_matrix[np.ix_(rows_to_select,)]
            
        
        # Use the final centroids as initial centroids for a new run
        kmeans = KMeans(n_clusters=optimal_k, init=initial_centers, n_init=1)
        cluster_assignments = kmeans.fit_predict(latency_matrix)       
        # Print the results
        Data = {}
        Data_ = {}
        for i in range(optimal_k):
            nodes_in_cluster = np.where(cluster_assignments == i)[0]
            data20 = []
            A_ = []
            B_ = []
            for item in nodes_in_cluster:
                data20.append(data0[item])
                
                A_.append(float(data0[item]['latitude']))
                B_.append(float(data0[item]['longitude']))
            
            
            Data_['Region'+str(i+1)] = [B_,A_]
            Data['Region'+str(i+1)] = data20




        return Data
    def Interpolated_NYM(self):
       

        with open('Interpolated_NYM_250_DEC_2023.json') as File:
            data_list = json.load(File)
            
        europe = []
        asia = []
        north_america = []
        south_america = []
        africa = []
        australia = []
    
        for node in data_list:
            latitude = float(node.get('latitude', 0))
            longitude = float(node.get('longitude', 0))
    
            if 35.5 <= latitude <= 71.5 and -25 <= longitude <= 40:  # Europe
                europe.append(node)
            elif 10 <= latitude <= 60 and 25 <= longitude <= 180:  # Asia
                asia.append(node)
            elif 7 <= latitude <= 83 and -172 <= longitude <= -34:  # North America
                north_america.append(node)
            elif -56 <= latitude <= 15 and -122 <= longitude <= -35:  # South America
                south_america.append(node)
            elif -35 <= latitude <= 37 and -25 <= longitude <= 52:  # Africa
                africa.append(node)
            elif -42 <= latitude <= -10 and 112 <= longitude <= 153:  # Australia
                australia.append(node)
    
        # Plotting on a globe map
        plt.figure(figsize=(12, 8))
        m = Basemap(projection='mill', llcrnrlat=-60, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, resolution='c')
        m.drawcoastlines()
    
        def plot_nodes(nodes, label, color):
            lats = [float(node.get('latitude', 0)) for node in nodes]
            lons = [float(node.get('longitude', 0)) for node in nodes]
            x, y = m(lons, lats)
            m.scatter(x, y, label=label, color=color)
    
        plot_nodes(europe, 'Europe', 'blue')
        plot_nodes(asia, 'Asia', 'green')
        plot_nodes(north_america, 'North America', 'red')
        plot_nodes(south_america, 'South America', 'purple')
        plot_nodes(africa, 'Africa', 'orange')
        plot_nodes(australia, 'Australia', 'cyan')
    

        Data_ = {}
        Data_['Region1'] = europe
        Data_['Region2'] = north_america
        return Data_           
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    def classify_and_plot(self):

        
        
        with open('D:/Approach3/Made_up_data.json') as json_file: 
        
            data_list = json.load(json_file) # Your list of dictionaries
        

        europe = []
        asia = []
        north_america = []
        south_america = []
        africa = []
        australia = []
    
        for node in data_list:
            latitude = node.get('latitude', 0)
            longitude = node.get('longitude', 0)
    
            if 35.5 <= latitude <= 71.5 and -25 <= longitude <= 40:  # Europe
                europe.append(node)
            elif 10 <= latitude <= 60 and 25 <= longitude <= 180:  # Asia
                asia.append(node)
            elif 7 <= latitude <= 83 and -172 <= longitude <= -34:  # North America
                north_america.append(node)
            elif -56 <= latitude <= 15 and -122 <= longitude <= -35:  # South America
                south_america.append(node)
            elif -35 <= latitude <= 37 and -25 <= longitude <= 52:  # Africa
                africa.append(node)
            elif -42 <= latitude <= -10 and 112 <= longitude <= 153:  # Australia
                australia.append(node)
    


        Data_ = {}
        Data_['Region1'] = europe
        Data_['Region2'] = asia
        return Data_
  
    
    
    
    def find_central_point(self,data):
        # Check if the input data is not empty
        if not data:
            return None
    
        # Calculate average longitude and latitude
        total_longitude = 0
        total_latitude = 0
        num_points = len(data)
    
        for point in data:
            total_longitude += point['longitude']
            total_latitude += point['latitude']
    
        average_longitude = total_longitude / num_points
        average_latitude = total_latitude / num_points
    
        # Find the dictionary with the closest longitude and latitude to the averages
        central_point = min(data, key=lambda point: abs(point['longitude'] - average_longitude) + abs(point['latitude'] - average_latitude))
    
        return central_point       




    def make_data(self):
       
        DF  = Dataset(self.data_type,self.N,self.Goal,self.CN,self.G)        
        DF.RIPE()        
        Data_set, Client_Data , GW_Data = DF.PLOT_New_dataset()
        self.Data_set = Data_set
        self.Clients  = Client_Data
        self.GW_Data      = GW_Data
        self.close_data = {}
       
   
    def data_interface(self):
        a = [(20,80),(-10,33) , (0,100),(33,100), (25,50),(-100,-50), (-40,25),(-100,-50)]
        b = ['Europe','Asia','North America','South America','Global']
   
        
        with open('cleaned_up_ripe_data_removed_negative_vals_2.json') as json_file: 
        
            data0 = json.load(json_file)
            
        Data = {}
        
        for i in range(int(len(a)/2)):
            
            Location = []
            Continental_Data = []
            l1 = a[2*i][0]
            u1 = a[2*i][1]
            
            l2 = a[2*i+1][0]
            u2 = a[2*i+1][1]            
            
            
            A = []
            B = []
                
            for j in range(len(data0)):
                if l1 <data0[j]['latitude'] < u1 and l2 <data0[j]['longitude']< u2:
                    Continental_Data.append(data0[j])
                    
                    A.append(data0[j]['latitude'])
                    B.append(data0[j]['longitude'])
            Location.append(B)
            Location.append(A)            
            Data['Location'+ b[i]] = Location
            Data['Data'+b[i]] = Continental_Data
        
        P = {}
        for item in b:
            if item == 'Global':
                pass
            else:
                P[item] = self.find_central_point(Data['Data'+item])
                Data['Data_'+item] = [point for point in Data['Data'+item] if point != P[item]]
                
                
            
        
        
        Regions  = {}
        for rr in b:
            if not rr=='Global':
                Regions[rr] = []
        G_Data = Data['Data_'+ b[0]]+Data['Data_'+ b[1]]+Data['Data_'+ b[2]]+Data['Data_'+ b[3]]
        
        Data['Data_'+b[4]] = G_Data
        
        
        for term in Data['Data_'+'Global']:
            Minimum = 100000
            for region in b:
                if not region == 'Global':
                    delay = self.delay_measure(P[region],term)
                    if delay<Minimum:
                        REGION = region
                        Minimum = delay
            Regions[REGION].append(term)
                        
        for i in range(len(Regions)):
            Regions[b[i]].append(P[b[i]])
            Data['Data'+b[i]] = Regions[b[i]]
            
        
        
        
        
        
        
        G_Data = Data['Data'+ b[0]]+Data['Data'+ b[1]]+Data['Data'+ b[2]]+Data['Data'+ b[3]]
        
        Data['Data'+b[4]] = G_Data      
        
        
        
        G_Location1 = Data['Location'+ b[0]][0] + Data['Location'+ b[1]][0]+Data['Location'+ b[2]][0]+Data['Location'+ b[3]][0]
        
        G_Location2 = Data['Location'+ b[0]][1] + Data['Location'+ b[1]][1]+Data['Location'+ b[2]][1]+Data['Location'+ b[3]][1]
         
        
        Data['Location'+b[4]] = [G_Location1,G_Location2]
        
        
        
        self.Data_Generation = Data
        
        self.Limit_N = {}
        for item in b:
            self.Limit_N[item] = len(Data['Data'+item])
        
        
    def delay_measure(self,A,B):
        I_key = B['i_key']
        In_Latency = A['latency_measurements'][str(I_key)]
        delay_distance = int(In_Latency)
        if delay_distance == 0:
            delay_distance =1
        return delay_distance/2000

    def MixNet_Creation(self,name,W):
  

        data0 = self.Data_Clustering__['Region'+name]# Data of each regions created based on goegraphical partitions

        
        GateWays = {}
        Layer = {'Layer1':{},'Layer2':{}}
        
        List = []
        for i in range(4*W):
            r = int(4*W*np.random.rand(1)[0])
            while r in List:
                r = int(4*W*np.random.rand(1)[0])
            List.append(r)
            
            
        B = [ data0[item]  for item in List if List.index(item) > (W-1)]

   


        
        self.close_data['Region'+name] =   B               
        for i in range(W):
            ID1 = List[i]
            for j in range(W,2*W):
                
                ID2 = List[j] 
                I_key = data0[ID2]['i_key']
                
                In_Latency = data0[ID1]['latency_measurements'][str(I_key)]
                delay_distance = self.Latency_reader_(In_Latency,self.TYPE)
                GateWays['G'+str(i+1) +'PM'+str(1+j-W)] = delay_distance/2000
            

        for k in range(2):
            for i in range(k*W,(k+1)*W):
                ID1 = List[i+W]
                for j in range((k+1)*W,(k+2)*W):
                    ID2 = List[j+W]   
                    I_key = data0[ID2]['i_key']
                    In_Latency = data0[ID1]['latency_measurements'][str(I_key)]
                    delay_distance = self.Latency_reader_(In_Latency,self.TYPE)             
                    Layer['Layer'+str(k+1)]['PM'+str(i+1) +'PM'+str(j+1)]= delay_distance/2000   
        
        return GateWays,Layer,List
        
  

    def Global_Capture(self,W,R):
        regions = ['1','2']
        
        
        w1 = int(len(R[regions[0]])/4)
        w2 = int(len(R[regions[1]])/4)
        
      
        ww = {}
        www = [w1,w2]
        i =0
        for region in regions:
            ww[region] = www[i]
            i = i+1
                
        
        name = 'Global'
        self.W = W
        

        data0 = self.Data_Clustering__
            
        
    
        GateWays = {}
        GW_data ={}
        Layers_data = {}
        for i in range(3):
            Layers_data['Layer'+str(i+1)] = []
        for region in regions:
            GW_data[region] = []
            GateWays[region] = {}
            
        
        Layer = {'Layer1':{},'Layer2':{}}
        
        for region in regions:
            
            for k in range(4*ww[region]):
                
                if k//ww[region]==0:
                    
                    GW_data[region].append(data0['Region'+region][R[region][k]])
                else:
                    Layers_data['Layer'+str(k//ww[region])].append(data0['Region'+region][R[region][k]])
    
        for region in regions:
            
            for i in range(ww[region]):
                
                for j in range(self.W):
                    I_key = Layers_data['Layer1'][j]['i_key']
                    In_Latency = GW_data[region][i]['latency_measurements'][str(I_key)]
                    delay_distance = self.Latency_reader_(In_Latency,self.TYPE)
                    GateWays[region]['G'+str(i+1) +'PM'+str(1+j)] = delay_distance/2000
            

        for k in range(2):
            for i in range(self.W):
                for j in range(self.W):
                    I_key = Layers_data['Layer'+str(2+k)][j]['i_key']
                    In_Latency = Layers_data['Layer'+str(1+k)][i]['latency_measurements'][str(I_key)]
                    delay_distance = self.Latency_reader_(In_Latency,self.TYPE)              
                    Layer['Layer'+str(k+1)]['PM'+str(i+1+k*self.W) +'PM'+str(j+1+(k+1)*self.W)]= delay_distance/2000   
        return GateWays,Layer






    
    def Circles_Creation(self,Iterations,WW):
        regions = ['1','2']
        Names = ['1','2'] +['Global']
        Tau = self.Var
        Tau[0] = 0.01

        DATA = {}     
        xxxx1 = 0
        xxxx2 = 0
        for I in range(Iterations):
            
            DATA_New = {}
            Data_Sim = self.PreProcessing(WW)      
            for name in Names:
                W = WW[name]

                
                if name == 'Global':
                    DATA1_ = {}
                    for region in regions:
                        w = WW[region]
                        W = WW['Global']
                        DATA1 = {}
                
                        for tau in Tau:
                            DATA2 = {}
                            for i in range(w):
                                t3 = time.time()
                                DD1 = self.LARMIX(Data_Sim['Global'][region]['G'+str(i+1)],tau)
                                t4 = time.time()
                                self.s_l = t4 - t3+self.stt
                                DD2 = Data_Sim['Global'][region]['G'+str(i+1)]
                                DATA2['G'+str(i+1)] = [DD2,DD1]
                            for i in range(W):
                                DD3 = self.LARMIX(Data_Sim['Global'][region]['PM'+str(i+1)],tau)
                                DD4 = Data_Sim['Global'][region]['PM'+str(i+1)]
                                DATA2['PM'+str(i+1)] = [DD4,DD3]                    
                            for i in range(W,2*W):
                                DD1 = self.LARMIX(Data_Sim['Global'][region]['PM'+str(i+1)],tau)
                                DD2 = Data_Sim['Global'][region]['PM'+str(i+1)]
                                DATA2['PM'+str(i+1)] = [DD2,DD1] 
                            DATA1['tau'+str(tau)] = DATA2
                        DATA1_[region] = DATA1
                    DATA_New[name] = DATA1_
                else:
                    DATA1 = {}
                    if name == '1':
                        x11 = time.time()
                    else:
                        x22 = time.time()
                    for tau in Tau:
                        DATA2 = {}
                        for i in range(W):
                            DD1 = self.LARMIX(Data_Sim[name]['G'+str(i+1)],tau)
                            DD2 = Data_Sim[name]['G'+str(i+1)]
                            DATA2['G'+str(i+1)] = [DD2,DD1]
                            DD3 = self.LARMIX(Data_Sim[name]['PM'+str(i+1)],tau)
                            DD4 = Data_Sim[name]['PM'+str(i+1)]
                            DATA2['PM'+str(i+1)] = [DD4,DD3]                    
                        for i in range(W,2*W):
                            DD1 = self.LARMIX(Data_Sim[name]['PM'+str(i+1)],tau)
                            DD2 = Data_Sim[name]['PM'+str(i+1)]
                            DATA2['PM'+str(i+1)] = [DD2,DD1] 
                        DATA1['tau'+str(tau)] = DATA2
                    if name == '1':
                        x11 = time.time() - x11
                    else:
                        x22 = time.time() - x22

                    DATA_New[name] = DATA1
            


                
            xxxx1 = xxxx1 + x11
            xxxx2 = xxxx2 + x22
            DATA['Iteration'+str(I)] = DATA_New
            DATA['Iteration'+str(I)]['Close'] = self.close_data
            
        self.time_is_money = [xxxx1,xxxx2]    
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
            
    

    def Simulator(self,corrupted_Mix,Mix_Dict,W,GG): 
   
        Mixes = [] #All mix nodes
        GateWays = {}
        env = simpy.Environment()    #simpy environment
        capacity=[]
        for j in range(3*W):# Generating capacities for mix nodes  
            c = simpy.Resource(env,capacity = self.CAP)
            capacity.append(c)           
        for i in range(3*W):#Generate enough instantiation of mix nodes  
            ll = i +1
            X = corrupted_Mix['PM%d' %ll]
            x = Mix(env,'M%02d' %i,capacity[i],X,self.N_target,self.d1)
            Mixes.append(x)
        
 
        for i in range(GG):#Generate enough instantiation of GateWays  
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
  
    def PreProcessing(self,W):

        Data_R = {}
        regions = ['1','2']
        Names =  regions + ['Global']
        for region in regions:
            Data_R[region] = {}
        

        Names_Data = {} 
        for name in Names:
            
            if name == 'Global':
                GateWays,Layers = self.Global_Capture(W[name],Data_R)
                
            else:
                t1 = time.time()
                GateWays,Layers,R = self.MixNet_Creation(name,W[name])
                t2 = time.time()
                self.stt = t2 - t1
                Data_R[name] = R

         
        
            if name == 'Global':
                Global_Data = {}
                
                for region in regions:
                    
                    m = W[region]
                    MM = W['Global']
                    data = {}
                    
                    
                    for i in range(m):
                        aa = []
                        for j in range(MM):
                            aa.append(GateWays[region]['G'+str(i+1)+'PM'+str(j+1)])
                        data['G'+str(i+1)] = aa
             
                    for i in range(MM):
                        aa = []
                        for j in range(MM,2*MM):
                            aa.append(Layers['Layer1']['PM'+str(i+1)+'PM'+str(j+1)])
                        data['PM'+str(i+1)] = aa               
            
                    for i in range(MM,2*MM):
                        aa = []
                        for j in range(2*MM,3*MM):
                            aa.append(Layers['Layer2']['PM'+str(i+1)+'PM'+str(j+1)])
                        data['PM'+str(i+1)] = aa
                    Global_Data[region] = data
                Names_Data[name] = Global_Data
            
            else:
                m = W[name]
                data = {}
                
                
                for i in range(m):
                    aa = []
                    for j in range(m):
                        aa.append(GateWays['G'+str(i+1)+'PM'+str(j+1)])
                    data['G'+str(i+1)] = aa
         
                for i in range(m):
                    aa = []
                    for j in range(m,2*m):
                        aa.append(Layers['Layer1']['PM'+str(i+1)+'PM'+str(j+1)])
                    data['PM'+str(i+1)] = aa               
        
                for i in range(m,2*m):
                    aa = []
                    for j in range(2*m,3*m):
                        aa.append(Layers['Layer2']['PM'+str(i+1)+'PM'+str(j+1)])
                    data['PM'+str(i+1)] = aa
                Names_Data[name] = data
        return Names_Data
    
    
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
    
    def make_T(self,G2,G3):

        #g1 = np.matrix(G1)
        g2 = np.matrix(G2)
        g3 = np.matrix(G3)
        return g2.dot(g3)
        
    def Analytic_Entropy(self,Data,name,W,Yes = False):
        self.W = W
        threshold = 1/self.W

        Tau = self.Var
        Tau[0] =0.01
        Entropy_ = []
        Load_ = []
        for i in range(len(Data)):
            if name == 'Global':
                
                data = Data['Iteration'+str(i)][name]['1']
            else:
                
            
                data = Data['Iteration'+str(i)][name]
            
            EEE = []
            LL_ = []
            for alpha in Tau:
                #Gamma11 = []
                Gamma21 = []
                Gamma31 = []

                #for j in range(self.W):
                    #Gamma11.append(data['tau'+str(alpha)]['G'+str(j+1)][1])
                
                    
                for j in range(self.W):
                    Gamma21.append(data['tau'+str(alpha)]['PM'+str(j+1)][1])                   
                for j in range(self.W,2*(self.W)):
                    Gamma31.append(data['tau'+str(alpha)]['PM'+str(j+1)][1])
                
                LIST_LOAD = self.filter_matrix_entries(Gamma21, threshold)+self.filter_matrix_entries(Gamma31, threshold)
                max_load = (np.sum(LIST_LOAD))/(len(LIST_LOAD))
                LL_.append(max_load)
                T_0 = self.make_T(Gamma21,Gamma31)
                E_0 = self.Entropy_Transformation(T_0)
                EEE.append(E_0)
            Entropy_.append(EEE)
            Load_.append(LL_)
                 
        Matrix = np.transpose(np.matrix(Entropy_)).tolist()       
        
        Med_ = Med(Matrix,50)
        Matrix_ = np.transpose(np.matrix(Load_)).tolist()       
        
        med_ = Med(Matrix_,50)
        if Yes:
            return Med_,med_
        else:
            return Med_
                
                
            
                    
                
                    
                    
                    
                    
                    
                    
                    
                    
                    
                
                
    
    def Latency_Med(self,List):

        a = np.transpose(np.matrix(List))
        return Med(a.tolist())
    

    
    def percentile_from_probabilities(self,Factors, Delays, percentile):

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


            

    def Analytic_Latency_(self,Data1,Routing):

        U__ = []


        for Iteration in range(len(Data1)):
            
            Data = Data1['Iteration'+str(Iteration)]
            U = []

            for II in range(6):
                alpha = II/5

                
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
                U.append(Y1+Y2+Y3)                               
                                
                                
                           
                              
                        
                              
                                


            U__.append(U)
        output = []
        for i in range(6):
            List = []
            
            for j in range(len(U__)):
                
                List = List + U__[j][i]
            output.append(List)
            
                
                
            
            #Output = self.Latency_Med(U__)   
            

        return  (np.matrix(Med(output,self.PP[0]))*3).tolist()[0], (np.matrix(Med(output,self.PP[1]))*3).tolist()[0]                        
                                


    def GW_Policy(self,Data,M):

        data = {}
        
        Name = ['Europe','Asia','North America']
        for name in Name:
            s = 0
            a = []
            for i in range(M):
                if Data['G'+str(i+1)] == name:
                    s = s +1
                    a.append(1)
                else:
                    a.append(0)
            b = (np.matrix(a)/s).tolist()[0]
            data[name] = b
        return data
            

    def Analytic_Latency_Global(self,Data1,W):
        self.W = W['Global']
        Name = ['1','2']
        
        
        Tau = self.Var
        Tau[0] = 0.01

        U__ = []


        for Iteration in range(len(Data1)):
            
            Data = Data1['Iteration'+str(Iteration)]['Global']
            #dist = self.GW_Policy(Data['S'],W)
            UU = {}
            for name in Name:
                M1 = W[name]
                
                
                U = []
    
                for alpha in Tau:
    
                    S = 0
                    for i in range(M1):
                        for j in range(self.W):
                            for k in range(self.W,2*self.W):
                                for z in range(2*self.W,3*self.W):
                                    
                                    x = Data[name]['tau'+str(alpha)]['G'+str(i+1)][0][j] + Data[name]['tau'+str(alpha)]['PM'+str(j+1)][0][k-self.W]+Data[name]['tau'+str(alpha)]['PM'+str(k+1)][0][z-2*self.W]
                                    y = (1/M1)*Data[name]['tau'+str(alpha)]['G'+str(i+1)][1][j]*Data[name]['tau'+str(alpha)]['PM'+str(j+1)][1][k-self.W]*Data[name]['tau'+str(alpha)]['PM'+str(k+1)][1][z-2*self.W]
                                    S = S + x*y
                    U.append(S)
                UU[name] =U
            U__.append(UU)
                        
                        
        U_1  = []
        U_2  = []

                    
        for item in U__:
            U_1.append(item[Name[0]])
            U_2.append(item[Name[1]])                                           

        output1 = np.transpose(np.matrix(U_1)).tolist()
        output2 = np.transpose(np.matrix(U_2)).tolist()           
      
        output = {'1':Med(output1,self.PP[0]),'2':Med(output2,self.PP[0])}
        
        return  output                       
                                

        
            

    def Analytic_Latency_normal(self,Data1,W,name):
        self.W = W
        
        Tau = self.Var
        Tau[0] = 0.01

        U__ = []


        for Iteration in range(len(Data1)):
            
            Data = Data1['Iteration'+str(Iteration)][name]
            U = []

            for alpha in Tau:

                S = 0
                for i in range(self.W):
                    for j in range(self.W):
                        for k in range(self.W,2*self.W):
                            for z in range(2*self.W,3*self.W):
                                
                                x = Data['tau'+str(alpha)]['G'+str(i+1)][0][j] + Data['tau'+str(alpha)]['PM'+str(j+1)][0][k-self.W]+Data['tau'+str(alpha)]['PM'+str(k+1)][0][z-2*self.W]
                                y = (1/self.W)*Data['tau'+str(alpha)]['G'+str(i+1)][1][j]*Data['tau'+str(alpha)]['PM'+str(j+1)][1][k-self.W]*Data['tau'+str(alpha)]['PM'+str(k+1)][1][z-2*self.W]
                                S = S + x*y
                U.append(S)
            U__.append(U)
                        
                        
                        
                    
                              
                                



        output = np.transpose(np.matrix(U__)).tolist()

            

        return  Med(output,self.PP[0]), Med(output,self.PP[1])                        
                                



    def E2E(self,e2e,Iteration,Name_):

        File_name = Name_        
        import os
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))  
        Names = ['Global','1','2']
        Names_ = ['1','2']
        WW = {'Global':52,'1':39,'2':13}
        
        Dictionaries = self.Circles_Creation(Iteration,WW)

        IT = 'Iteration'
        Latency1 = []
        Latency2 = []
        Entropy1 = []
        MyData = Dictionaries
        df = {}
        
        for names in Names:
            
            if names == 'Global':
                Y = self.Analytic_Latency_Global(MyData,WW)
            else:
                
                o1,o2 = self.Analytic_Latency_normal(MyData,WW[names],names)
                Latency1.append(o1)
                Latency2.append(o2)

            
            Entropy1.append(self.Analytic_Entropy(MyData,names,WW[names]))
            
        df['Analytical_Entropy_Global'] = Entropy1[0]

        df['Analytical_Entropy_'+(Names[1])] = Entropy1[1]     
        df['Analytical_Entropy_'+(Names[2])] = Entropy1[2]
        
        Mix_delays = {}
        for name in Names:
            if name == 'Global':
                Mix_delays[name+'1'] = []
                Mix_delays[name+'2'] = []
            else:
                Mix_delays[name] = []
            
        for item in Y:
            for ii in range(len(Y[item])):
                Mix_delays['Global'+item].append((e2e-Y[item][ii])/3)
            df['Link_delays'+'Global'+item] = Y[item]
            df['Mixing_delays'+ 'Global'+item] = Mix_delays['Global'+item]
        
        for jj in range(len(Latency1)):
            for ii in range(len(Latency1[0])):
                Mix_delays[str(jj+1)].append((e2e-Latency1[jj][ii])/3)
            df['Link_delays'+str(jj+1)] = Latency1[jj]
            df['Mixing_delays'+str(jj+1)]  = Mix_delays[str(jj+1)]
            
            
        #print(df)
#################################################################################################        
    ##################Simulation with No adverserial node#####################################
        #################################################################
        ##########################
        Latency_alpha_Global1 = []
        Latency_alpha_Global1_T = []    
        Entropy_alpha_Global1= []
        Latency_alpha_1 = []
        Latency_alpha_1_T = []    
        Entropy_alpha_1 = []

        corrupted_Mix = {}
        Var =  [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        X_name = ['Region1','Region2','Global1','Global2']
        x_name = X_name[2]
        xx_value = 0.5        
###########################################Global1  ##############################  
        C_numbers = 0
        for j in Var:
            alpha = j    
            self.d1 = Mix_delays['Global'+'1'][C_numbers]
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            #print(output__['Iteration'+str(1)]['CN'][0])
            for i in range(len(Dictionaries)):
                
                            

                Mix_Dict = {}
                if x_name[0] == 'G':
                    self.G = 39
                    
                    #self.G = int(xx_value*len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(1)][0]))
                #else:
                    #self.G = len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(1)][0])
                    
                
                #print(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)])
                for I in range(self.G):

                    if x_name[0] == 'G':
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(I+1)]
                        
                    else: 
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(I+1)]
    


                if x_name[0] == 'G':

                    self.W = int(len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(1)][0]))
                else:
                    self.W = int(len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(1)][0]))                  
                    
                for Counter__ in range(3*self.W):
                    corrupted_Mix['PM'+str(Counter__+1)] = False
                    
                for J in range(2*self.W):

                    if x_name[0] == 'G':
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(J+1)]
    
                    else:
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(J+1)]
                #print(self.G,self.W,'Hi')
                self.dd = self.d2
                self.d2 = self.d2*(52/52)
                Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Mix_Dict,self.W,self.G)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_alpha_Global1.append(End_to_End_Latancy_Vector)
            Latency_alpha_Global1_T.append(End_to_End_Latancy_Vector_T)
            Entropy_alpha_Global1.append(Message_Entropy_Vector)
            C_numbers = C_numbers + 1
            
        Sim_Latency = {}
        Sim_Entropy = {}

        for item in Names:
            if item == 'Global':
                Sim_Latency['Global'+'1'] = []
                Sim_Latency['Global'+'2'] = []
                Sim_Entropy['Global'+'1'] = []
                Sim_Entropy['Global'+'2'] = []
            else:
                Sim_Latency[item] = []
                Sim_Entropy[item] = []
        for ii in range(len(self.Var)):
            Sim_Latency['Global'+str(1)].append(np.mean(Latency_alpha_Global1[ii]))
            Sim_Entropy['Global'+str(1)].append(np.mean(Entropy_alpha_Global1[ii]))
        df['Simulated_Latency'+'Global'+str(1)] = Sim_Latency['Global'+str(1)]
        df['Simulated_Entropy'+'Global'+str(1)] = Sim_Entropy['Global'+str(1)]
        
            
                
            
    












###########################################Region1 ############################## 
        x_name = '1'
        C_numbers = 0
        for j in Var:
            self.d1 = Mix_delays['1'][C_numbers]
            alpha = j       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            #print(output__['Iteration'+str(1)]['CN'][0])
            for i in range(len(Dictionaries)):
                            

                Mix_Dict = {}
                if x_name[0] == 'G':
                    self.G = 39
                    
                    #self.G = int(xx_value*len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(1)][0]))
                #else:
                    #self.G = len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(1)][0])
                    
                
                #print(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)])
                for I in range(self.G):

                    if x_name[0] == 'G':
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(I+1)]
                        
                    else: 
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(I+1)]
    


                if x_name[0] == 'G':

                    self.W = int(len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(1)][0]))
                else:
                    self.W = int(len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(1)][0]))                  
                for Counter__ in range(3*self.W):
                    corrupted_Mix['PM'+str(Counter__+1)] = False                    
                
                for J in range(2*self.W):

                    if x_name[0] == 'G':
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(J+1)]
    
                    else:
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(J+1)]
                #print(self.G,self.W,'Hi')
                self.d2 = self.dd
                self.d2 = self.d2*(52/48)
                Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Mix_Dict,self.W,self.G)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_alpha_1.append(End_to_End_Latancy_Vector)
            Latency_alpha_1_T.append(End_to_End_Latancy_Vector_T)
            Entropy_alpha_1.append(Message_Entropy_Vector)
            
            C_numbers = C_numbers +1

        for ii in range(len(self.Var)):
            Sim_Latency[str(1)].append(np.mean(Latency_alpha_1[ii]))
            Sim_Entropy[str(1)].append(np.mean(Entropy_alpha_1[ii]))
        df['Simmulatted_Latency'+str(1)] = Sim_Latency[str(1)]
        df['Simmulatted_Entropy'+str(1)] = Sim_Entropy[str(1)]        
        #print(Sim_Latency,Sim_Entropy)

#################################################################################################        
    ##################Simulation with No adverserial node#####################################
        #################################################################
        ##########################
        Latency_alpha_Global2 = []
        Latency_alpha_Global2_T = []    
        Entropy_alpha_Global2= []
        Latency_alpha_2 = []
        Latency_alpha_2_T = []    
        Entropy_alpha_2 = []

        corrupted_Mix = {}
        Var =  [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
        X_name = ['Region1','Region2','Global1','Global2']
        x_name = X_name[3]
        xx_value = 0.5        
###########################################Global2  ##############################  
        C_numbers = 0
        for j in Var:
            alpha = j    
            self.d1 = Mix_delays['Global'+'2'][C_numbers]
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            #print(output__['Iteration'+str(1)]['CN'][0])
            for i in range(len(Dictionaries)):
                
                            

                Mix_Dict = {}
                if x_name[0] == 'G':
                    self.G =13
                    
                    #self.G = int(xx_value*len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(1)][0]))
                #else:
                    #self.G = len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(1)][0])
                    
                
                #print(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)])
                for I in range(self.G):

                    if x_name[0] == 'G':
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(I+1)]
                        
                    else: 
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(I+1)]
    


                if x_name[0] == 'G':

                    self.W = int(len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(1)][0]))
                else:
                    self.W = int(len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(1)][0]))                  
                    
                for Counter__ in range(3*self.W):
                    corrupted_Mix['PM'+str(Counter__+1)] = False
                    
                for J in range(2*self.W):

                    if x_name[0] == 'G':
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(J+1)]
    
                    else:
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(J+1)]
                #print(self.G,self.W,'Hi')
                self.d2 = self.dd
                self.d2 = self.d2*(52/52)
                Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Mix_Dict,self.W,self.G)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_alpha_Global2.append(End_to_End_Latancy_Vector)
            Latency_alpha_Global2_T.append(End_to_End_Latancy_Vector_T)
            Entropy_alpha_Global2.append(Message_Entropy_Vector)
            C_numbers = C_numbers + 1
    
        for ii in range(len(self.Var)):
            Sim_Latency['Global'+str(2)].append(np.mean(Latency_alpha_Global2[ii]))
            Sim_Entropy['Global'+str(2)].append(np.mean(Entropy_alpha_Global2[ii]))
        df['Simulated_Latency'+'Global'+str(2)] = Sim_Latency['Global'+str(2)]
        df['Simulated_Entropy'+'Global'+str(2)] = Sim_Entropy['Global'+str(2)]











###########################################Region2 ############################## 
        x_name = '2'
        C_numbers = 0
        for j in Var:
            self.d1 = Mix_delays['2'][C_numbers]
            alpha = j       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            #print(output__['Iteration'+str(1)]['CN'][0])
            for i in range(len(Dictionaries)):
                            

                Mix_Dict = {}
                if x_name[0] == 'G':
                     self.G =13
                    
                    #self.G = int(xx_value*len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(1)][0]))
                #else:
                    #self.G = len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(1)][0])
                    
                
                #print(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)])
                for I in range(self.G):

                    if x_name[0] == 'G':
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(I+1)]
                        
                    else: 
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(I+1)]
    


                if x_name[0] == 'G':

                    self.W = int(len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(1)][0]))
                else:
                    self.W = int(len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(1)][0]))                  
                for Counter__ in range(3*self.W):
                    corrupted_Mix['PM'+str(Counter__+1)] = False                    
                
                for J in range(2*self.W):

                    if x_name[0] == 'G':
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(J+1)]
    
                    else:
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(J+1)]
                #print(self.G,self.W,'Hi')
                self.d2 = self.dd
                self.d2 = self.d2*(52/40)
                Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Mix_Dict,self.W,self.G)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_alpha_2.append(End_to_End_Latancy_Vector)
            Latency_alpha_2_T.append(End_to_End_Latancy_Vector_T)
            Entropy_alpha_2.append(Message_Entropy_Vector)
            
            C_numbers = C_numbers +1

        for ii in range(len(self.Var)):
            Sim_Latency[str(2)].append(np.mean(Latency_alpha_2[ii]))
            Sim_Entropy[str(2)].append(np.mean(Entropy_alpha_2[ii]))
        df['Simulated_Latency'+str(2)] = Sim_Latency[str(2)]
        df['Simulated_Entropy'+str(2)] = Sim_Entropy[str(2)]
        
        df['Tau'] = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]


        with open('Results'+'/E2E_Limit_RM.json','w') as file:
            json.dump(df,file)



#######################################################################################

    def filter_matrix_entries(self,matrix, threshold):
        

        # Convert the matrix to numpy array for easier manipulation
        matrix = np.array(matrix)
        
        # Boolean indexing to filter entries greater than the specified value
        filtered_entries = matrix[matrix > threshold]
        
        # Convert the filtered entries to a list
        filtered_list = filtered_entries.tolist()
        
        return filtered_list

#######################################################################################

#######################################################################################

        
    def FCP_Greedy(self,data,G_mean,Type):

        C = FCP_Mix(data,self.Adversary_Budget)
        if Type=='Random':
            C_nodes,FCP = C.C_random(G_mean)
        elif Type=='Close':
            C_nodes,FCP = C.Close_knit_nodes(G_mean,self.T_data)            
        elif Type=='Greedy':
            C_nodes,FCP = C.Greedy_For_Fairness(G_mean) 
        return [C_nodes, FCP]
    
    def FCP(self,Iteration,Name_):
        import os
        File_name = Name_        
        import os
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))  
        Names = ['Global','1','2']
        Names_ = ['1','2']
        WW = {'Global':52,'1':39,'2':13}
        
        Dictionaries = self.Circles_Creation(Iteration,WW)

        IT = 'Iteration'

        Var = self.Var
        Methods = ['Random','Close','Greedy']
        output__ = {}
        for ii in range(Iteration):
            output__[IT+str(ii)] = {}
            Dict_FCP = {'output_Random_FCP':{},'output_Greedy_FCP':{},'output_Close_FCP':{}}
            Dict_CN = {'output_Random_CN':{},'output_Greedy_CN':{},'output_Close_CN':{}}
   
            
            for item in Names:
                
                if not item == 'Global':
            
                    for term in self.Var:
                        #print(term)
                        G_dist = []
                        for k in range(WW[item]):
                            #print(Dictionaries[IT+str(ii)]['G'+str(k+1)])
                            
                            G_dist.append(Dictionaries[IT+str(ii)][item]['tau'+str(term)]['G'+str(k+1)][1])
                            
                        G_matrix = np.matrix(G_dist)
                        G_mean = np.mean(G_matrix,axis=0).tolist()[0]
                        Input = {}
                        for i in range(2*WW[item]):
                            Input['PM' + str(i+1)] = Dictionaries[IT+str(ii)][item]['tau'+str(term)]['PM'+str(i+1)][1]
                        Dict_output = {}    
                        for method in Methods:
                            if not method == 'Close':
                                Dict_output['output'+method] = self.FCP_Greedy(Input,G_mean,method)
                                
    
                            else:
                                self.T_data = Dictionaries[IT+str(ii)]['Close']['Region'+item]
                                Dict_output['output'+method] = self.FCP_Greedy(Input,G_mean,method)
    
                            
                            Dict_FCP['output_' + method +'_FCP'][str(term)+item ]= Dict_output['output'+method][1]

                            Dict_CN['output_' + method +'_CN'][str(term)+item ]= Dict_output['output'+method][0]
                
                
                
                
                
                
                
                elif item == 'Global':
                    
                    for element in Names_:
                        
                        for term in self.Var:
                            #print(term)
                            G_dist = []
                            for k in range(WW[element]):
                                #print(Dictionaries[IT+str(ii)]['G'+str(k+1)])

                                G_dist.append(Dictionaries[IT+str(ii)][item][element]['tau'+str(term)]['G'+str(k+1)][1])
                                
                            G_matrix = np.matrix(G_dist)
                            G_mean = np.mean(G_matrix,axis=0).tolist()[0]
                            Input = {}
                            for i in range(2*WW[item]):
                                Input['PM' + str(i+1)] = Dictionaries[IT+str(ii)][item][element]['tau'+str(term)]['PM'+str(i+1)][1]
                            Dict_output = {}    
                            for method in Methods:
                                if not method == 'Close':
                                    Dict_output['output'+method] = self.FCP_Greedy(Input,G_mean,method)
                                    
        
                                else:
                                    self.T_data = []
                                    for parts in Names_:
                                        
                                        self.T_data.append(Dictionaries[IT+str(ii)]['Close']['Region'+parts])
                                    Dict_output['output'+method] = self.FCP_Greedy(Input,G_mean,method)
        
                                
                                Dict_FCP['output_' + method +'_FCP'][str(term)+item+element ]= Dict_output['output'+method][1]
                                Dict_CN['output_' + method +'_CN'][str(term)+item+element ]= Dict_output['output'+method][0]                        


            output__[IT+str(ii)]['CN'] = [Dict_CN['output_Random_CN'],Dict_CN['output_Close_CN'],Dict_CN['output_Greedy_CN'] ]
            output__[IT+str(ii)]['FCP'] = [Dict_FCP['output_Random_FCP'],Dict_FCP['output_Close_FCP'],Dict_FCP['output_Greedy_FCP']]
        IDs = ['1','2','Global1','Global2']
        AVE_FCP = {}
        for m_name in Methods:
            AVE_FCP[m_name] = {}
        for counter in range(len(Methods)):
            for item in IDs:
                for term in self.Var:
                    A = []
                    for j in range(Iteration):
                        b = output__[IT+str(j)]['FCP'][counter][str(term)+item]

                        A.append(b)
                    A_matrix = np.matrix(A)
                    A_mean = np.mean(A,axis=0)
                    AVE_FCP[Methods[counter]][str(term)+item] = A_mean
                    

        File_name = Name_        
        import os
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))  
        Names_ = ['Global1','Global2','1','2'] 
        Y = {} 
        for m_name in Methods:
            Y[m_name] = {}
        for m_name in Methods:
            for name in Names_:
                Y[m_name][name] = []
                for term in self.Var:
                    Y[m_name][name].append(AVE_FCP[m_name][str(term)+name])
    

        X_L = r'$\tau $'
        Y_t = 'Throughput'
        Y_L = "Fraction of Corrupted Paths"
        M_ = ['Random','Single Location','Worst Case']
        
        #DD = ['Uniform','LARMIX'+ r'$\tau=$' + str(0.9),'Proportional','LARMIX'+r'$\tau=$'+str(0.6),'LARMIX'+r'$\tau=$'+str(0.3)] 
        #DD = ['Uniform','Proportional','LARMIX'+ r'$\tau=$' + str(0.3),'LARMIX'+ r'$\tau=$' + str(0.6),'LARMIX'+ r'$\tau=$' + str(0.9)]
        DD = ['Global1','Global2','Region1','Region2']        
        Description = []
        Alpha = self.Var
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
        PLT_Random.scatter_line(True,0.05)


        PLT_Close = PLOT(Alpha,Y_C,DD,X_L,Y_L,FCP_Close)
        PLT_Close.scatter_line(True,0.1)


        PLT_Greedy = PLOT(Alpha,Y_G,DD,X_L,Y_L,FCP_Greedy)
        PLT_Greedy.scatter_line(True,0.35)               
                
        self.Simulation_FCP = output__
        FCP_Dicts = {'FCP':Y,'FCP_Sim':output__}

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
        ######Assign the following before the simulation###########
        Var =  [0.01,0.2,0.4,0.6,0.8,1]
        X_name = ['Region1','Region2','Global1','Global2']
        x_name = X_name[2]
        xx_value = 0.5#Becouse we have to region with the same size otherwise G_reginal/sum(G_all_regions)##############

###########################################Global1 + Random ##############################  
        
        for j in Var:
            alpha = j       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            #print(output__['Iteration'+str(1)]['CN'][0])
            for i in range(len(Dictionaries)):
                corrupted_Mix = output__['Iteration'+str(i)]['CN'][0][str(alpha)+'Global1']
                            

                Mix_Dict = {}
                if x_name[0] == 'G':
                    
                    self.G = int(xx_value*len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(1)][0]))
                else:
                    self.G = len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(1)][0])
                    
                
                #print(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)])
                for I in range(self.G):

                    if x_name[0] == 'G':
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(I+1)]
                        
                    else: 
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(I+1)]
    


                if x_name[0] == 'G':

                    self.W = int(len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(1)][0]))
                else:
                    self.W = int(len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(1)][0]))                  
                    
                
                for J in range(2*self.W):

                    if x_name[0] == 'G':
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(J+1)]
    
                    else:
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(J+1)]
                #print(self.G,self.W,'Hi')
                self.d2 = self.d2*(52/52)
                Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Mix_Dict,self.W,self.G)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_alpha_Uniform.append(End_to_End_Latancy_Vector)
            Latency_alpha_Uniform_T.append(End_to_End_Latancy_Vector_T)
            Entropy_alpha_Uniform.append(Message_Entropy_Vector)
    



###########################################Global1 + Close ##############################   
        for j in Var:
            alpha = j       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            #print(output__['Iteration'+str(1)]['CN'][0])
            for i in range(len(Dictionaries)):
                corrupted_Mix = output__['Iteration'+str(i)]['CN'][1][str(alpha)+'Global1']
                            

                Mix_Dict = {}
                if x_name[0] == 'G':
                    
                    self.G = int(xx_value*len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(1)][0]))
                else:
                    self.G = len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(1)][0])
                    
                
                #print(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)])
                for I in range(self.G):

                    if x_name[0] == 'G':
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(I+1)]
                        
                    else: 
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(I+1)]
    


                if x_name[0] == 'G':

                    self.W = int(len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(1)][0]))
                else:
                    self.W = int(len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(1)][0]))                  
                    
                
                for J in range(2*self.W):

                    if x_name[0] == 'G':
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(J+1)]
    
                    else:
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(J+1)]
                #print(self.G,self.W,'Hi')
                self.d2 = self.d2*(52/52)
                Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Mix_Dict,self.W,self.G)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_alpha_Fair.append(End_to_End_Latancy_Vector)
            Latency_alpha_Fair_T.append(End_to_End_Latancy_Vector_T)
            Entropy_alpha_Fair.append(Message_Entropy_Vector)



###########################################Global1 + Greedy ##############################   
        for j in Var:
            alpha = j       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            #print(output__['Iteration'+str(1)]['CN'][0])
            for i in range(len(Dictionaries)):
                corrupted_Mix = output__['Iteration'+str(i)]['CN'][2][str(alpha)+'Global1']
                            

                Mix_Dict = {}
                if x_name[0] == 'G':
                    
                    self.G = int(xx_value*len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(1)][0]))
                else:
                    self.G = len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(1)][0])
                    
                
                #print(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)])
                for I in range(self.G):

                    if x_name[0] == 'G':
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(I+1)]
                        
                    else: 
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(I+1)]
    


                if x_name[0] == 'G':

                    self.W = int(len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(1)][0]))
                else:
                    self.W = int(len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(1)][0]))                  
                    
                
                for J in range(2*self.W):

                    if x_name[0] == 'G':
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(J+1)]
    
                    else:
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(J+1)]
                #print(self.G,self.W,'Hi')
                self.d2 = self.d2*(52/52)
                Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Mix_Dict,self.W,self.G)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_alpha_LARMIX.append(End_to_End_Latancy_Vector)
            Latency_alpha_LARMIX_T.append(End_to_End_Latancy_Vector_T)
            Entropy_alpha_LARMIX.append(Message_Entropy_Vector)
        
            
##################################################################################
##################################################################################
            

        labels = [0.0,0.2,0.4,0.6,0.8,1]

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


        dics = json.dumps(df)
        with open(File_name + '/'+ 'FCP' +'Sim.json','w') as df_sim:
            json.dump(dics,df_sim)                
        
        


#################################################################################################        
    ##################Simulation with No adverserial node#####################################
        #################################################################
        ##########################
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
        
###########################################Global1  ##############################   
        for j in Var:
            alpha = j       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            #print(output__['Iteration'+str(1)]['CN'][0])
            for i in range(len(Dictionaries)):
                
                            

                Mix_Dict = {}
                if x_name[0] == 'G':
                    
                    self.G = int(xx_value*len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(1)][0]))
                else:
                    self.G = len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(1)][0])
                    
                
                #print(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)])
                for I in range(self.G):

                    if x_name[0] == 'G':
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(I+1)]
                        
                    else: 
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(I+1)]
    


                if x_name[0] == 'G':

                    self.W = int(len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(1)][0]))
                else:
                    self.W = int(len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(1)][0]))                  
                    
                for Counter__ in range(3*self.W):
                    corrupted_Mix['PM'+str(Counter__+1)] = False
                    
                for J in range(2*self.W):

                    if x_name[0] == 'G':
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(J+1)]
    
                    else:
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(J+1)]
                #print(self.G,self.W,'Hi')
                self.d2 = self.d2*(52/52)
                Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Mix_Dict,self.W,self.G)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_alpha_Uniform.append(End_to_End_Latancy_Vector)
            Latency_alpha_Uniform_T.append(End_to_End_Latancy_Vector_T)
            Entropy_alpha_Uniform.append(Message_Entropy_Vector)
    



###########################################Region1 ############################## 
        x_name = '1'
        for j in Var:
            alpha = j       
            End_to_End_Latancy_Vector = []
            End_to_End_Latancy_Vector_T = []
            Message_Entropy_Vector = []            
            #print(output__['Iteration'+str(1)]['CN'][0])
            for i in range(len(Dictionaries)):
                            

                Mix_Dict = {}
                if x_name[0] == 'G':
                    
                    self.G = int(xx_value*len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(1)][0]))
                else:
                    self.G = len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(1)][0])
                    
                
                #print(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)])
                for I in range(self.G):

                    if x_name[0] == 'G':
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['G'+str(I+1)]
                        
                    else: 
                        Mix_Dict['G'+str(I+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['G'+str(I+1)]
    


                if x_name[0] == 'G':

                    self.W = int(len(Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(1)][0]))
                else:
                    self.W = int(len(Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(1)][0]))                  
                for Counter__ in range(3*self.W):
                    corrupted_Mix['PM'+str(Counter__+1)] = False                    
                
                for J in range(2*self.W):

                    if x_name[0] == 'G':
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)]['Global'][x_name[-1]]['tau'+str(alpha)]['PM'+str(J+1)]
    
                    else:
                        Mix_Dict['PM'+str(J+1)] = Dictionaries['Iteration'+str(i)][x_name]['tau'+str(alpha)]['PM'+str(J+1)]
                #print(self.G,self.W,'Hi')
                self.d2 = self.d2*(52/50)
                Latencies, Latencies_T,ENT = self.Simulator(corrupted_Mix,Mix_Dict,self.W,self.G)
                End_to_End_Latancy_Vector =  End_to_End_Latancy_Vector + Latencies
                End_to_End_Latancy_Vector_T =  End_to_End_Latancy_Vector_T + Latencies_T
                Message_Entropy_Vector = Message_Entropy_Vector + ENT  
                    
            Latency_alpha_Fair.append(End_to_End_Latancy_Vector)
            Latency_alpha_Fair_T.append(End_to_End_Latancy_Vector_T)
            Entropy_alpha_Fair.append(Message_Entropy_Vector)


            

        labels = [0.0,0.2,0.4,0.6,0.8,1]

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


        dics = json.dumps(df)
        with open(File_name + '/'+ 'FCP' +'Sim_simple.json','w') as df_sim:
            json.dump(dics,df_sim)        
        
        
        


#################################################################################################        
                
        
        
        
        
        
    def FCP_Budget(self,Iteration,Name_,Budget):
        
        self.Adversary_Budget = Budget

        File_name = Name_        
        import os
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))  
        Names = ['Global','1','2']
        Names_ = ['1','2']
        WW = {'Global':52,'1':39,'2':13}
        
        Dictionaries = self.Circles_Creation(Iteration,WW)

        IT = 'Iteration'

        Var = [0.6]
        Methods = ['Greedy']
        output__ = {}
        for ii in range(Iteration):
            output__[IT+str(ii)] = {}
            Dict_FCP = {'output_Greedy_FCP':{}}
            Dict_CN = {'output_Greedy_CN':{}}
   
            
            for item in Names:
                
                if not item == 'Global':
            
                    for term in self.Var:
                        #print(term)
                        G_dist = []
                        for k in range(WW[item]):
                            #print(Dictionaries[IT+str(ii)]['G'+str(k+1)])
                            
                            G_dist.append(Dictionaries[IT+str(ii)][item]['tau'+str(term)]['G'+str(k+1)][1])
                            
                        G_matrix = np.matrix(G_dist)
                        G_mean = np.mean(G_matrix,axis=0).tolist()[0]
                        Input = {}
                        for i in range(2*WW[item]):
                            Input['PM' + str(i+1)] = Dictionaries[IT+str(ii)][item]['tau'+str(term)]['PM'+str(i+1)][1]
                        Dict_output = {}    
                        for method in Methods:
                            if not method == 'Close':
                                Dict_output['output'+method] = self.FCP_Greedy(Input,G_mean,method)
                                
    
                            else:
                                self.T_data = Dictionaries[IT+str(ii)]['Close']['Region'+item]
                                Dict_output['output'+method] = self.FCP_Greedy(Input,G_mean,method)
    
                            
                            Dict_FCP['output_' + method +'_FCP'][str(term)+item ]= Dict_output['output'+method][1]

                            Dict_CN['output_' + method +'_CN'][str(term)+item ]= Dict_output['output'+method][0]
                
                
                
                
                
                
                
                elif item == 'Global':
                    
                    for element in Names_:
                        
                        for term in self.Var:
                            #print(term)
                            G_dist = []
                            for k in range(WW[element]):
                                #print(Dictionaries[IT+str(ii)]['G'+str(k+1)])

                                G_dist.append(Dictionaries[IT+str(ii)][item][element]['tau'+str(term)]['G'+str(k+1)][1])
                                
                            G_matrix = np.matrix(G_dist)
                            G_mean = np.mean(G_matrix,axis=0).tolist()[0]
                            Input = {}
                            for i in range(2*WW[item]):
                                Input['PM' + str(i+1)] = Dictionaries[IT+str(ii)][item][element]['tau'+str(term)]['PM'+str(i+1)][1]
                            Dict_output = {}    
                            for method in Methods:
                                if not method == 'Close':
                                    Dict_output['output'+method] = self.FCP_Greedy(Input,G_mean,method)
                                    
        
                                else:
                                    self.T_data = []
                                    for parts in Names_:
                                        
                                        self.T_data.append(Dictionaries[IT+str(ii)]['Close']['Region'+parts])
                                    Dict_output['output'+method] = self.FCP_Greedy(Input,G_mean,method)
        
                                
                                Dict_FCP['output_' + method +'_FCP'][str(term)+item+element ]= Dict_output['output'+method][1]
                                Dict_CN['output_' + method +'_CN'][str(term)+item+element ]= Dict_output['output'+method][0]                        


            output__[IT+str(ii)]['CN'] = [Dict_CN['output_Greedy_CN'] ]
            output__[IT+str(ii)]['FCP'] = [Dict_FCP['output_Greedy_FCP']]
        IDs = ['1','2','Global1','Global2']
        AVE_FCP = {}
        for m_name in Methods:
            AVE_FCP[m_name] = {}
        for counter in range(len(Methods)):
            for item in IDs:
                for term in [0.6]:
                    A = []
                    for j in range(Iteration):
                        b = output__[IT+str(j)]['FCP'][counter][str(term)+item]

                        A.append(b)
                    A_matrix = np.matrix(A)
                    A_mean = np.mean(A,axis=0)
                    AVE_FCP[Methods[counter]][str(term)+item] = A_mean
                    

        File_name = Name_        
        import os 
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))  
        Names_ = ['Global1','Global2','1','2'] 
        Y = {} 
        for m_name in Methods:
            Y[m_name] = {}
        for m_name in Methods:
            for name in Names_:
                Y[m_name][name] = []
                for term in [0.6]:
                    Y[m_name][name].append(AVE_FCP[m_name][str(term)+name])
                    
                    
                    
        


        self.Simulation_FCP = output__
        FCP_Dicts = {'FCP':Y,'FCP_Sim':output__}

        with open(File_name+'/FCP_Data.json','w') as file:
            json.dump(FCP_Dicts,file)
        
        
        
        
        
        
    def EL_Analysis1(self,Name_,Iteration,K_Cluster):

        File_name = Name_        
        import os   
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))  
        Names = ['Global','1','2']
        WW = {'Global':52,'1':39,'2':13}
        Latency1 = []
        Latency2 = []
        Entropy1 = []
        YourData = {}
        Latency1.append([0])

        t1 = time.time()
        MyData = self.Circles_Creation(Iteration,WW)

        YourData = MyData
        
        for names in Names:
            
            if names == 'Global':
                Y = self.Analytic_Latency_Global(MyData,WW)
            else:
                
                o1,o2 = self.Analytic_Latency_normal(MyData,WW[names],names)
                Latency1.append(o1)
                Latency2.append(o2)

            
            Entropy1.append(self.Analytic_Entropy(MyData,names,WW[names]))

        Frac = []
        LatencY = []
        for I in range(len(Entropy1)):
            if I==0:
                for Term in ['1','2']:
                    
                    Latency100 = np.matrix(Y[Term])
                    LatencY.append(Y[Term])
                    Entropy100 = np.matrix(Entropy1[I])
                    f1 = Entropy100/Latency100
                    Frac.append(f1.tolist()[0])
            else:
                
                Latency100 = np.matrix(Latency1[I])
                LatencY.append(Latency1[I])
                Entropy100 = np.matrix(Entropy1[I])
                f1 = Entropy100/Latency100
                Frac.append(f1.tolist()[0])
                
        Data_saving = {'Alpha':self.Var,
            'Latency':LatencY,'Entropy':Entropy1,'Frac':Frac}
  
        file_name = Name_+'/data_Analytic.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(Data_saving, file)      



        X_L = r'$\tau$'
        Y_t = 'Entropy/Latency'
        Y_E = "Entropy (bits)"
        Y_L = 'Latency (sec)'
        D = ['Region1','Region2']
        D1 = ['Global']+D
        
        DD = ['Region1_Global','Region2_Global']
        DDD = DD+D
        Alpha = self.Var

            
        Name_Entropy = File_name + '/' + str(self.PP[0])+'Entropy.png'
        Name_Latency = File_name + '/'+ str(self.PP[0]) + 'Latency.png'
        Name_Latency_ = File_name + '/'+ str(self.PP[1]) + 'Latency.png'
        Name_t = File_name + '/'+ 'Throughput.png'            
        PLT_E = PLOT(Alpha,Entropy1,D1,X_L,Y_E,Name_Entropy)
        PLT_E.scatter_line(True,7.5)
        


        PLT_t = PLOT(Alpha,Frac,DDD,X_L,Y_t,Name_t)
        PLT_t.scatter_line(True,200)
        
        
        
        PLT_L1 = PLOT(Alpha,LatencY,DDD,X_L,Y_L,Name_Latency)

        PLT_L1.scatter_line(True,0.4)
        '''
        PLT_L1 = PLOT(Alpha,Latency2,D,X_L,Y_L,Name_Latency_)

        PLT_L1.scatter_line(True,0.4)  
        '''
        
        Data_saving = {'Alpha':Alpha,
            'Latency':LatencY,'Entropy':Entropy1,'Frac':Frac}
        
        file_name = Name_+'/data_Analytic.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(Data_saving, file)





            







    def EL_Analysis1_RIPE(self,Name_,Iteration,K_Cluster):
   
        File_name = Name_        
        import os
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))  
        Names = ['Global','1','2']
        WW = {'Global':52,'1':39,'2':13}
        Latency1 = []
        Latency2 = []
        Entropy1 = []
        YourData = {}
        Latency1.append([0])

        t1 = time.time()
        MyData = self.Circles_Creation(Iteration,WW)
        #print(time.time()-t1)
        YourData = MyData
        
        for names in Names:
            
            if names == 'Global':
                Y = self.Analytic_Latency_Global(MyData,WW)
            else:
                
                o1,o2 = self.Analytic_Latency_normal(MyData,WW[names],names)
                Latency1.append(o1)
                Latency2.append(o2)

            
            Entropy1.append(self.Analytic_Entropy(MyData,names,WW[names]))

        Frac = []
        LatencY = []
        for I in range(len(Entropy1)):
            if I==0:
                for Term in ['1','2']:
                    
                    Latency100 = np.matrix(Y[Term])
                    LatencY.append(Y[Term])
                    Entropy100 = np.matrix(Entropy1[I])
                    f1 = Entropy100/Latency100
                    Frac.append(f1.tolist()[0])
            else:
                
                Latency100 = np.matrix(Latency1[I])
                LatencY.append(Latency1[I])
                Entropy100 = np.matrix(Entropy1[I])
                f1 = Entropy100/Latency100
                Frac.append(f1.tolist()[0])
                
        Data_saving = {'Alpha':self.Var,
            'Latency':LatencY,'Entropy':Entropy1,'Frac':Frac}
  
        file_name = Name_+'/data_Analytic.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(Data_saving, file)      


        X_L = r'$\tau$'
        Y_t = 'Entropy/Latency'
        Y_E = "Entropy (bits)"
        Y_L = 'Latency (sec)'
        D = ['Region1','Region2']
        D1 = ['Global']+D
        
        DD = ['Region1_Global','Region2_Global']
        DDD = DD+D
        Alpha = self.Var

            
        Name_Entropy = File_name + '/' + str(self.PP[0])+'Entropy.png'
        Name_Latency = File_name + '/'+ str(self.PP[0]) + 'Latency.png'
        Name_Latency_ = File_name + '/'+ str(self.PP[1]) + 'Latency.png'
        Name_t = File_name + '/'+ 'Throughput.png'            
        PLT_E = PLOT(Alpha,Entropy1,D1,X_L,Y_E,Name_Entropy)
        PLT_E.scatter_line(True,7.5)
        


        PLT_t = PLOT(Alpha,Frac,DDD,X_L,Y_t,Name_t)
        PLT_t.scatter_line(True,200)
        
        
        
        PLT_L1 = PLOT(Alpha,LatencY,DDD,X_L,Y_L,Name_Latency)

        PLT_L1.scatter_line(True,0.4)
        '''
        PLT_L1 = PLOT(Alpha,Latency2,D,X_L,Y_L,Name_Latency_)

        PLT_L1.scatter_line(True,0.4)  
        '''
        
        Data_saving = {'Alpha':Alpha,
            'Latency':LatencY,'Entropy':Entropy1,'Frac':Frac}
 
        file_name = Name_+'/data_Analytic.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(Data_saving, file)





            








    def EL_Analysis(self,Name_,Iteration,K_Cluster):

        File_name = Name_        
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))  
        Names = ['Global','Europe','Asia','North America','South America']
        WW = {'Global':50,'Asia':10,'North America':14,'South America':6,'Europe':70}
        
        
        
        Latency1 = []
        Latency2 = []
        Entropy1 = []
        YourData = {}
        Latency1.append([0])
            
        MyData = Sim.Circles_Creation(Iteration,WW)
        YourData = MyData
        
        for names in Names:
            
            if names == 'Global':
                Y = Sim.Analytic_Latency_Global(MyData,WW)
            else:
                
                o1,o2 = Sim.Analytic_Latency_normal(MyData,WW[names],names)
                Latency1.append(o1)
                Latency2.append(o2)

            
            Entropy1.append(self.Analytic_Entropy(MyData,names,WW[names]))

        Frac = []
        LatencY = []
        for I in range(len(Entropy1)):
            if I==0:
                for Term in ['E','A','NA','SA']:
                    
                    Latency100 = np.matrix(Y[Term])
                    LatencY.append(Y[Term])
                    Entropy100 = np.matrix(Entropy1[I])
                    f1 = Entropy100/Latency100
                    Frac.append(f1.tolist()[0])
            else:
                
                Latency100 = np.matrix(Latency1[I])
                LatencY.append(Latency1[I])
                Entropy100 = np.matrix(Entropy1[I])
                f1 = Entropy100/Latency100
                Frac.append(f1.tolist()[0])
                
        Data_saving = {'Alpha':self.Var,
            'Latency':LatencY,'Entropy':Entropy1,'Frac':Frac}

        file_name = Name_+'/data_Analytic.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(Data_saving, file)      


        X_L = r'$\tau$'
        Y_t = 'Entropy/Latency'
        Y_E = "Entropy (bits)"
        Y_L = 'Latency (sec)'
        D = ['EU','AS','NA']
        D1 = ['Global']+D
        
        DD = ['EU(Global)','AS(Global)','NA(Global)']
        DDD = DD+D
        Alpha = self.Var

            
        Name_Entropy = File_name + '/' + str(self.PP[0])+'Entropy.png'
        Name_Latency = File_name + '/'+ str(self.PP[0]) + 'Latency.png'
        Name_Latency_ = File_name + '/'+ str(self.PP[1]) + 'Latency.png'
        Name_t = File_name + '/'+ 'Throughput.png'            
        PLT_E = PLOT(Alpha,Entropy1,D1,X_L,Y_E,Name_Entropy)
        PLT_E.scatter_line(True,7.5)
        


        PLT_t = PLOT(Alpha,Frac,DDD,X_L,Y_t,Name_t)
        PLT_t.scatter_line(True,200)
        
        
        
        PLT_L1 = PLOT(Alpha,LatencY,DDD,X_L,Y_L,Name_Latency)

        PLT_L1.scatter_line(True,0.4)
        '''
        PLT_L1 = PLOT(Alpha,Latency2,D,X_L,Y_L,Name_Latency_)

        PLT_L1.scatter_line(True,0.4)  
        '''
        
        Data_saving = {'Alpha':Alpha,
            'Latency':LatencY,'Entropy':Entropy1,'Frac':Frac}
     
        file_name = Name_+'/data_Analytic.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(Data_saving, file)
            
    def LARMix_EXP(self,W):
        pass



    def data_117_NYM_(self):
        #117_nodes_latency_December_2023_cleaned_up_9_no_intersection_1


   
        
        with open('D:/Approach3/117_nodes_latency_December_2023_cleaned_up_9_no_intersection_1.json') as json_file: 
        
            data_list = json.load(json_file) # Your list of dictionaries
        
        #Plotting data on the globe
        europe, asia, north_america, south_america, africa, australia = Sim.classify_and_plot_NYM(data_list)
        Names = ['Europe','Asia','North America','South America','Global']
        Data = {}
        Data['Data'+Names[0]] = europe
        Data['Data'+Names[2]] = north_america
        Data['Data'+Names[4]] = europe+north_america
        
        self.Data_Generation = Data





        
        
        
        
        
    def Load_Analysis(self,Name_,Iteration,K_Cluster):

        File_name = Name_        
        import os
        if not os.path.exists(File_name):
            os.mkdir(os.path.join('', File_name))  
        Names = ['Global','1','2']
        WW = {'Global':52,'1':39,'2':13}
        Loads = []
            
        MyData = self.Circles_Creation(Iteration,WW)
        YourData = MyData
        
        for names in Names:
            

            
            E_1 , L_1 = self.Analytic_Entropy(MyData,names,WW[names],True)
            
            Loads.append(L_1)

        D = ['EU','AS','NA']
        D1 = ['Global']+D

                
        Data_saving = {'Alpha':self.Var,
            'Loads':Loads}
 
        file_name = Name_+'/Loads_Analytic.pkl'
        with open(file_name, 'wb') as file:
            pickle.dump(Data_saving, file)      

     

        X_L = r'$\tau$'
        Y_L = 'Over Loads Percentage'
        
        DD = ['Region1_Global','Region2_Global']
        Alpha = self.Var

            
        Name_Load = File_name + '/' + str(self.PP[0])+'Load.png'
        
        PLT_E = PLOT(Alpha,Loads,D1,X_L,Y_L,Name_Load)
        PLT_E.scatter_line(True,1)
        





    def Time_Analysis(self,Name_,Iteration,K_Cluster):
 
        Names = ['Global','1','2']
        WW = {'Global':52,'1':39,'2':13}

        MyData = self.Circles_Creation(Iteration,WW)
        
        return self.time_is_money[0]/(len(self.Var)*Iteration),self.time_is_money[1]/(len(self.Var)*Iteration)























