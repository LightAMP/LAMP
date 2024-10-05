from Multi_Circles import CircularMixNet_MC
from Single_Circle import CircularMixNet_SC

from Regional import Regional_MixNet


class LAMP(object):
    
    def __init__(self):
        self.W = 60
        self.N = 3*self.W
        self.num_gateways = self.W
        self.delay1 = 0.05
        self.delay2 = 0.0001/2
        self.Capacity = 10000000000000000000000000000000000000000000000000000000000000000
        self.H_N = round(self.N/3)
        self.rate = 100
        self.num_targets = 200
        self.Iterations = 20
        self.run = 0.5
        self.Percentile = [50,95]        
        
        
    def data_initialization(self,Iterations):
        import os         
        if not os.path.exists('Results'):
            os.mkdir(os.path.join('', 'Results'))   
        from data0 import MakeData
        
        
        Class = MakeData(self.W, Iterations)
        
        Class.Common_Data()
        
        
    def SC_Baselines(self,Name_File,Iterations):
        #from Multi_Circles import CircularMixNet

        Class_SC = CircularMixNet_SC(self.num_targets,Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.H_N,self.N,self.rate,self.num_gateways,self.Percentile,Name_File) 


        #Sim.EL_Analysis('BaseLine'+str(W)+'Nodes_per_Layers',40,True)
        
        Class_SC.EL_Analysis(Name_File,Iterations,True)        
        
        #################################BaseLine####Approach2#################################################
        #######################################################################################################
        
        
        from Plot import PLOT
        
        
        X_L = r'Radius $r$ (ms)'
        Y_t = 'Entropy/Latency'
        Y_E = "Entropy (bits)"
        Y_L = 'Latency (sec)'
        
        D = ['Uniform',r'LARMix $\tau=0.9$','Proportional',r'LARMix $\tau=0.6$',r'LARMix $\tau=0.3$']
            
        Name_E_A2 = 'Results' + '/' +'Fig_5a.png'
        
        Name_L_A2 = 'Results' + '/' +'Fig_5d.png'
        
        Name_EL_A2 = 'Results' + '/' +'Fig_5g.png'
        
        ######################################################################################################
        ######################################################################################################
        import pickle
        File_name = Name_File
        with open(File_name + '/'+'Analytical.pkl', 'rb') as file:
            # Serialize and save your data to the file
            a = pickle.load(file)
            
        
            
        Tau = a['Alpha']
        for i in range(len(Tau)):
            Tau[i] = Tau[i]*1000
        
        Entropy = a['Entropy']
        x1 = Entropy[1]
        x2 = Entropy[2]
        Entropy[1] = x2
        Entropy[2] = x1
        PLT_E = PLOT(Tau,Entropy,D,X_L,Y_E,Name_E_A2)
        PLT_E.scatter_line(True,10)
        
        
        
        
        Latency = a['Latency']
        D = ['Uniform','Proportional',r'LARMix $\tau=0.9$',r'LARMix $\tau=0.6$',r'LARMix $\tau=0.3$']
        L1 = Latency[1]
        L2 = Latency[2]
        Latency[1] = L2
        Latency[2] = L1
        PLT_L = PLOT(Tau,Latency,D,X_L,Y_L,Name_L_A2,True,False,False,True)
        Color1 = PLT_L.colors[1]
        Color2 = PLT_L.colors[2]
        #print(Color2,Color1)
        PLT_L.colors[1] = Color2
        PLT_L.colors[2] = Color1
        p2 = PLT_L.Line_style[1] 
        p3 = PLT_L.Line_style[2]
        PLT_L.Line_style[1]  = p3
        PLT_L.Line_style[2] =  p2
        
        PLT_L.scatter_line(True,0.18)

                
                
        
        
    def MC_Baselines(self,Name_File,Iterations):
        #from Multi_Circles import CircularMixNet

        Class_MC = CircularMixNet_MC(self.num_targets,Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.H_N,self.N,self.rate,self.num_gateways,self.Percentile,Name_File) 


        #Sim.EL_Analysis('BaseLine'+str(W)+'Nodes_per_Layers',40,True)
        
        Class_MC.EL_Analysis(Name_File,Iterations,True)
    
        #################################BaseLine####Approach1#################################################
        
        
        
        from Plot import PLOT
        
        
        X_L = r'Radius $r$ (ms)'
        Y_t = 'Entropy/Latency'
        Y_E = "Entropy (bits)"
        Y_L = 'Latency (sec)'
        
        D = ['Uniform',r'LARMix $\tau=0.9$','Proportional',r'LARMix $\tau=0.6$',r'LARMix $\tau=0.3$']
            
        Name_E_A1 = 'Results' + '/' +'Fig_5b.png'
        
        Name_L_A1 = 'Results' + '/' +'Fig_5e.png'
        
        Name_EL_A1 = 'Results' + '/' +'Fig_5h.png'
        
        ######################################################################################################
        ######################################################################################################
        import pickle
        File_name = Name_File
        with open(File_name + '/'+'Analytical.pkl', 'rb') as file:
            # Serialize and save your data to the file
            a = pickle.load(file)
            
        
            
        Tau = a['Alpha']
        Tau = [1/1000,7/1000,15/1000,30/1000,50/1000,100/1000]
        for i in range(len(Tau)):
            Tau[i] = Tau[i]*1000
        
        Entropy = a['Entropy']
        x1 = Entropy[1]
        x2 = Entropy[2]
        Entropy[1] = x2
        Entropy[2] = x1
        y = [[]]*5
        for i in range(len(Entropy)):
            y[i] = Entropy[i][:5]
        x = Tau[:5]
        
        
        PLT_E = PLOT(x,y,D,X_L,Y_E,Name_E_A1)
        PLT_E.scatter_line(True,10)
        
        
        
        
        Latency = a['Latency']
        y = [[]]*5
        for i in range(len(Entropy)):
            y[i] = Latency[i][:5]
        x = Tau[:5]
        PLT_L = PLOT(x,y,D,X_L,Y_L,Name_L_A1)
        PLT_L.scatter_line(True,0.18)
        





        
    def RM_Baselines(self,Name_File,Iterations):
        #from Multi_Circles import CircularMixNet

        Class_RM = Regional_MixNet(self.num_targets,Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.H_N,self.N,self.rate,self.num_gateways,self.Percentile,Name_File) 


        #Sim.EL_Analysis('BaseLine'+str(W)+'Nodes_per_Layers',40,True)
        
        Class_RM.EL_Analysis1(Name_File,Iterations,2)  
        
        #################################BaseLine####Approach3#################################################
        ######################################################################################################
        ######################################################################################################
        
        
        from Plot import PLOT
        
        
        X_L = r'Randomness $\tau$'
        Y_t = 'Entropy/Latency'
        Y_E = "Entropy (bits)"
        Y_L = 'Latency (sec)'
        
        D = ['Global Mixnet','EU Mixnet','NA Mixnet']
        
            
        Name_E_A3 = 'Results' + '/' +'Fig_5c.png'
        
        Name_L_A3 = 'Results' + '/' +'Fig_5f.png'
        
        Name_EL_A3 = 'Results' + '/' +'Fig_5i.png'
        
        ######################################################################################################
        ######################################################################################################
        import pickle
        File_name = Name_File
        with open(File_name + '/'+'data_Analytic.pkl', 'rb') as file:
            # Serialize and save your data to the file
            a = pickle.load(file)
            
        
            
        Tau = a['Alpha']
        
        
        Entropy = a['Entropy']
        
        PLT_E = PLOT(Tau,Entropy,D,X_L,Y_E,Name_E_A3)
        PLT_E.scatter_line(True,10)
        
        
        D = ['NA Mixnet + NA Clients','Global Mixnet + NA Clients','Global Mixnet + EU Clients','EU Mixnet + EU Clients']
        
        Latency = a['Latency']
        PLT_t = PLOT(Tau,Latency,D,X_L,Y_L,Name_L_A3)
        
        
        #print(PLT_t.colors)
        
        
        f1 = Latency[0] 
        f2 = Latency[1]
        f3 = Latency[2]
        f4 = Latency[3]
        
        Latency[0]  = f4
        Latency[1]  = f2
        Latency[2]  = f1
        Latency[3]  = f3
        
        
        c1 = PLT_t.colors[0] 
        c2 = PLT_t.colors[1] 
        c3 = PLT_t.colors[2] 
        c4 = PLT_t.colors[3] 
        c5 = PLT_t.colors[4]
        PLT_t.colors[0] = c3
        PLT_t.colors[1] = c5
        PLT_t.colors[2] = c1
        PLT_t.colors[3] = c2
        #print(PLT_t.colors)
        
        
        p1 = PLT_t.Line_style[0] 
        p2 = PLT_t.Line_style[1] 
        p3 = PLT_t.Line_style[2] 
        p4 = PLT_t.Line_style[3] 
        p5 = PLT_t.Line_style[4] 
        PLT_t.Line_style[0] = p3
        PLT_t.Line_style[1] = p5
        PLT_t.Line_style[2] = p1
        PLT_t.Line_style[3] = p2
        
        
        
        PLT_t.scatter_line(True,0.25)
        
        
        
        





    def RM_Simulations_FCP(self,Name_File,Iterations):
        #from Multi_Circles import CircularMixNet

        Class_RM = Regional_MixNet(self.num_targets,Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.H_N,self.N,self.rate,self.num_gateways,self.Percentile,Name_File) 


        #Sim.EL_Analysis('BaseLine'+str(W)+'Nodes_per_Layers',40,True)
        
        Class_RM.FCP(Iterations,Name_File)

        #############################################################################################
        ####################################Simulations: Fig 6a and Fig6b##############################################
        #############################################################################################
        #############################################################################################
        import json
        
        with open(Name_File+'/FCPSim_simple.json','r') as file:
            data0 = json.loads(json.load(file))
             
                
        Name_E_S3 = 'Results' + '/' +'Fig_6a.png'
        
        Name_L_S3 = 'Results' + '/' +'Fig_6b.png'
        
        Aa = data0['Alpha']
        
        Y_E = [data0['Entropy_Uniform'],data0['Entropy_Fair']]
        Y_L = [data0['Latency_Uniform'],data0['Latency_Fair']]
        D = ['Global Mixnet + EU Clients','EU Mixnet + EU Clients']
        ##################################Plots##################################################           
        
        Y_Label_L = 'Latency (sec)'
        Y_Label_E = 'Entropy (bits)'
        X_Label = r'Randomness $\tau$ ' 
        
        from Plot import PLOT
        PLT = PLOT(Aa,Y_E,D,X_Label,Y_Label_E,Name_E_S3)
        
        PLT.Box_Plot(13,True)
        
        
        PLT = PLOT(Aa,Y_L,D,X_Label,Y_Label_L,Name_L_S3)
        PLT.Box_Plot(0.8,True)

        ########################################################################################
        ###########################################Fig_8c######################################
        ########################################################################################

        
        
        from Plot import PLOT
        
        
        X_L = r'Randomness $\tau$ '
        
        Y_f = "FCP"
        
        
        D = ['NA Mixnet + NA Clients','EU Mixnet + EU Clients','Global Mixnet + EU Clients','Global Mixnet + NA Clients']
            
        Name_FCP_A3 = 'Results' + '/' +'Fig_8c.png'
        
        Name_FCP_S3 = 'Results' + '/' +'Fig_7c.png'
        
        
        
        ######################################################################################################
        ######################################################################################################
        import json
        File_name = Name_File+'/FCP_Data.json'
        with open(File_name , 'r') as file:
            # Serialize and save your data to the file
            a = json.load(file)
            
         
        
        
        
        Y_FCP = []
        for item in a['FCP']['Greedy']:
            Y_FCP.append(a['FCP']['Greedy'][item])
            
        y1 = Y_FCP[0]
        y2= Y_FCP[1]
        y3 = Y_FCP[2]
        y4 = Y_FCP[3]   
        
        Y_FCP[0] = y4
        Y_FCP[1] = y3
        Y_FCP[2] = y1
        Y_FCP[3] = y2
        
        
        Tau = [i/10 for i in range(11)]
        
        
        PLT_f = PLOT(Tau,Y_FCP,D,X_L,Y_f,Name_FCP_A3,True,False,False,True)
        PLT_f.colors = [PLT_f.colors[2],PLT_f.colors[1],PLT_f.colors[0],PLT_f.colors[4],PLT_f.colors[0]]
        PLT_f.Line_style = [PLT_f.Line_style[2],PLT_f.Line_style[1],PLT_f.Line_style[0],PLT_f.Line_style[4],PLT_f.Line_style[0]]
        
        PLT_f.scatter_line(True,0.45)
        
        



    
    def SC_Simulation_FCP(self,Name_File,Iterations):
        #from Multi_Circles import CircularMixNet

        Class_SC = CircularMixNet_SC(self.num_targets,Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.H_N,self.N,self.rate,self.num_gateways,self.Percentile,Name_File) 


        #Sim.EL_Analysis('BaseLine'+str(W)+'Nodes_per_Layers',40,True)
        
        Class_SC.FCP(Iterations,Name_File,True)     


        
        ##########################################################################################
        #########################################################################################
        #################################FCP SC#################################################
        ##########################################################################################
        #########################################################################################
        ##########################################################################################
        #########################################################################################
        ##########################################################################################
        #########################################################################################
        
        
        from Plot import PLOT
        
        
        X_L = r'Radius $r$ (ms)'
        
        Y_f = "FCP"
        
        
        D = [r'LARMix $\tau=0.3$',r'LARMix $\tau=0.6$','Proportional',r'LARMix $\tau=0.9$','Uniform']
        
        Name_FCP_A2 = 'Results' + '/' +'Fig_8a.png'
        
        Name_FCP_S2 =  'Results' + '/' +'Fig_7a.png'
        
        
        
        ######################################################################################################
        ######################################################################################################
        import json
        File_name = Name_File+'/FCP_Data.json'
        with open(File_name , 'r') as file:
        # Serialize and save your data to the file
            a = json.load(file)
        
        Y_FCP = []
        for item in a['FCP']['Greedy']:
            Y_FCP.append(a['FCP']['Greedy'][item])
        
        Tau = [0.001,0.007,0.015,0.03,0.05]
        for i in range(len(Tau)):
            Tau[i] = Tau[i]*1000
        
        
        PLT_f = PLOT(Tau,Y_FCP,D,X_L,Y_f,Name_FCP_A2,True,False,False,True)
        PLT_f.colors = [PLT_f.colors[4],PLT_f.colors[3],PLT_f.colors[2],PLT_f.colors[1],PLT_f.colors[0]]
        PLT_f.Line_style = [PLT_f.Line_style[4],PLT_f.Line_style[3],PLT_f.Line_style[2],PLT_f.Line_style[1],PLT_f.Line_style[0]]
        
        PLT_f.scatter_line(True,0.4)
        

        
        


    def MC_Simulation_FCP(self,Name_File,Iterations):
        #from Multi_Circles import CircularMixNet

        Class_MC = CircularMixNet_MC(self.num_targets,Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.H_N,self.N,self.rate,self.num_gateways,self.Percentile,Name_File) 


        #Sim.EL_Analysis('BaseLine'+str(W)+'Nodes_per_Layers',40,True)
        
        Class_MC.FCP(Iterations,Name_File,True)     


        
        ##########################################################################################
        #########################################################################################
        #################################FCP SC#################################################
        ##########################################################################################
        #########################################################################################
        ##########################################################################################
        #########################################################################################
        ##########################################################################################
        #########################################################################################
        
        
        from Plot import PLOT
        
        
        X_L = r'Radius $r$ (ms)'
        
        Y_f = "FCP"
        
        
        D = [r'LARMix $\tau=0.3$',r'LARMix $\tau=0.6$','Proportional',r'LARMix $\tau=0.9$','Uniform']
        
        Name_FCP_A2 = 'Results' + '/' +'Fig_8b.png'
        
        Name_FCP_S2 =  'Results' + '/' +'Fig_7b.png'
        
        
        
        ######################################################################################################
        ######################################################################################################
        import json
        File_name = Name_File+'/FCP_Data.json'
        with open(File_name , 'r') as file:
        # Serialize and save your data to the file
            a = json.load(file)
        
        Y_FCP = []
        for item in a['FCP']['Greedy']:
            Y_FCP.append(a['FCP']['Greedy'][item][:5])
        
        Tau = [0.001,0.007,0.015,0.03,0.05]
        for i in range(len(Tau)):
            Tau[i] = Tau[i]*1000
        
        
        PLT_f = PLOT(Tau,Y_FCP,D,X_L,Y_f,Name_FCP_A2,True,False,False,True)
        PLT_f.colors = [PLT_f.colors[4],PLT_f.colors[3],PLT_f.colors[2],PLT_f.colors[1],PLT_f.colors[0]]
        PLT_f.Line_style = [PLT_f.Line_style[4],PLT_f.Line_style[3],PLT_f.Line_style[2],PLT_f.Line_style[1],PLT_f.Line_style[0]]
        
        PLT_f.scatter_line(True,0.4)
        

        

















