

# -*- coding: utf-8 -*-
"""
Common Dataset: This file creates configurations for mixnets that can be used across all approaches to ensure fair comparisons.
"""
import json
import numpy as np
class MakeData(object):
    
    def __init__(self,W,Iteration):
        
        self.W = W
        self.Iteration = Iteration
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
    def MixNet_Creation(self):  
        GateWays = {}
        Layer = {'Layer1':{},'Layer2':{}}
        GateWays__ = {}

    
        with open('Interpolated_NYM_250_DEC_2023.json') as json_file: 
        
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
                if not type(In_Latency) == float:
                    delay_distance = self.Latency_reader(In_Latency)
                else:
                    delay_distance = In_Latency
                if delay_distance == 0:
                    delay_distance =1
                GateWays['G'+str(i+1) +'PM'+str(1+j-self.W)] = abs(delay_distance)/2000
        for i in range(self.W):
            ID1 = List[i]
            for j in range(self.W,4*self.W):
                ID2 = List[j] 
                I_key = data0[ID2]['i_key']
                In_Latency = data0[ID1]['latency_measurements'][str(I_key)]
                if not type(In_Latency) == float:
                    delay_distance = self.Latency_reader(In_Latency)
                else:
                    delay_distance = In_Latency
                if delay_distance == 0:
                    delay_distance =1
                GateWays__['G'+str(i+1) +'PM'+str(1+j-self.W)] = abs(delay_distance)/2000
                        

        for k in range(2):
            for i in range(k*self.W,(k+1)*self.W):
                ID1 = List[i+self.W]
                for j in range((k+1)*self.W,(k+2)*self.W):
                    ID2 = List[j+self.W]   
                    I_key = data0[ID2]['i_key']
                    In_Latency = data0[ID1]['latency_measurements'][str(I_key)]
                    if not type(In_Latency) == float:
                        delay_distance = self.Latency_reader(In_Latency)
                    else:
                        delay_distance = In_Latency
                    if delay_distance == 0:
                        delay_distance =1                    
                    Layer['Layer'+str(k+1)]['PM'+str(i+1) +'PM'+str(j+1)]= abs(delay_distance)/2000                 
        self.GG = GateWays__           
        return GateWays,Layer 
    
    def Common_Data(self):
        Com_Data = {}
        Com_Data2 = {}
        Data_ = {}
        
        for i in range(self.Iteration):
            G,L = self.MixNet_Creation()
            
            Com_Data['Iteration'+str(i+1)] = [G,L,self.close_data]
            Data_['Iteration'+str(i+1)] = self.close_data
            Com_Data2['Iteration'+str(i+1)] = [self.GG,L,self.close_data]

        with open('LARMIX__.json','w') as file:
            json.dump(Com_Data,file)

        with open('LARMIX__2.json','w') as file:
            json.dump(Com_Data2,file)            

        with open('LARMIX.json','w') as file:
            json.dump(Data_,file)










          