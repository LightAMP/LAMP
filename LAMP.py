from Multi_Circles import CircularMixNet_MC
from Single_Circle import CircularMixNet_SC
import numpy as np
from Regional import Regional_MixNet

def polynomial_extrapolation(X, Y, x_extrapolate, degree=2):
    if len(X) != len(Y):
        raise ValueError("The lists X and Y must have the same length.")
    if len(X) < 2:
        raise ValueError("The lists X and Y must have at least two points for extrapolation.")
    
    # Sort the data points by X values
    sorted_points = sorted(zip(X, Y))
    X, Y = zip(*sorted_points)
    
    # Fit a polynomial of the specified degree to the data points
    coefficients = np.polyfit(X, Y, degree)
    polynomial = np.poly1d(coefficients)
    
    # Extrapolate the value
    y_extrapolated = polynomial(x_extrapolate)
    return y_extrapolated


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

        D = [r'LARMix $\tau=0.9$',r'LARMix $\tau=0.6$',r'LARMix $\tau=0.3$','Uniform','Proportional']
               
        L1 = Latency[1]
        L2 = Latency[2]
        Latency[1] = L2
        Latency[2] = L1
        Frac = []
        for j in range(len(D)):
            
            Frac.append([Entropy[j][i]/Latency[j][i] for i in range(len(Tau))])
        PLT_t = PLOT(Tau,Frac,D,X_L,Y_t,Name_EL_A2)
        
        f1 = Frac[0] 
        f2 = Frac[1]
        f3 = Frac[2]
        f4 = Frac[3]
        f5 = Frac[4]
        
        Frac[0]  = f2
        Frac[1]  = f4
        Frac[2]  = f5
        Frac[3]  = f1
        Frac[4]  = f3
        
        c1 = PLT_t.colors[0] 
        c2 = PLT_t.colors[1] 
        c3 = PLT_t.colors[2] 
        c4 = PLT_t.colors[3] 
        c5 = PLT_t.colors[4]
        PLT_t.colors[0] = c2
        PLT_t.colors[1] = c4
        PLT_t.colors[2] = c5
        PLT_t.colors[3] = c1
        PLT_t.colors[4] = c3
        
        
         
        
        
        
        p1 = PLT_t.Line_style[0] 
        p2 = PLT_t.Line_style[1] 
        p3 = PLT_t.Line_style[2] 
        p4 = PLT_t.Line_style[3] 
        p5 = PLT_t.Line_style[4] 
        PLT_t.Line_style[0] = p2
        PLT_t.Line_style[1] = p4
        PLT_t.Line_style[2] = p5
        PLT_t.Line_style[3] = p1
        PLT_t.Line_style[4] = p3
        
        
        
        PLT_t.scatter_line(True,270)                
                
        
        
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
        
        D = [r'LARMix $\tau=0.6$','Proportional',r'LARMix $\tau=0.9$','Uniform',r'LARMix $\tau=0.3$']
        #['royalblue','red','green','fuchsia','cyan','indigo','teal','lime','blue','black','orange','violet','lightblue']
               # self.Line_style = ['-',':','--','-.','--','-',':','--','-.','--']  
        Frac = []
        for j in range(len(D)):
            
            Frac.append([Entropy[j][i]/Latency[j][i] for i in range(len(Tau))])
            
        
        f1 = Frac[0] 
        f2 = Frac[1]
        f3 = Frac[2]
        f4 = Frac[3]
        f5 = Frac[4]
        
        Frac[0]  = f4
        Frac[1]  = f3
        Frac[2]  = f2
        Frac[3]  = f1
        Frac[4]  = f5
        x = Tau[:5]
        y = [[]]*5
        for i in range(len(Frac)):
            y[i] = Frac[i][:5]
        
        PLT_t = PLOT(x,y,D,X_L,Y_t,Name_EL_A1,True,False,False,True)
        
        
        c1 = PLT_t.colors[0] 
        c2 = PLT_t.colors[1] 
        c3 = PLT_t.colors[2] 
        c4 = PLT_t.colors[3] 
        c5 = PLT_t.colors[4]
        PLT_t.colors[0] = c4
        PLT_t.colors[1] = c3
        PLT_t.colors[2] = c2
        PLT_t.colors[3] = c1
        PLT_t.colors[4] = c5
        
        
         
        
        
        
        p1 = PLT_t.Line_style[0] 
        p2 = PLT_t.Line_style[1] 
        p3 = PLT_t.Line_style[2] 
        p4 = PLT_t.Line_style[3] 
        p5 = PLT_t.Line_style[4] 
        PLT_t.Line_style[0] = p4
        PLT_t.Line_style[1] = p3
        PLT_t.Line_style[2] = p2
        PLT_t.Line_style[3] = p1
        PLT_t.Line_style[4] = p5
        
        
        
        PLT_t.scatter_line(True,270)




        
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
        
        
        Latency = a['Latency']
        
        
        
        
        
        
        D = ['EU Mixnet + EU Clients','Global Mixnet + EU Clients','Global Mixnet + NA Clients','NA Mixnet + NA Clients']
        #['royalblue','red','green','fuchsia','cyan','indigo','teal','lime','blue','black','orange','violet','lightblue']
               # self.Line_style = ['-',':','--','-.','--','-',':','--','-.','--']
        Entropy_=[]
        for i in range(len(Entropy)):
            if i==0:
                Entropy_.append(Entropy[0])
                Entropy_.append(Entropy[0])
            else:
                Entropy_.append(Entropy[i])
        Frac = []
        for j in range(len(D)):
            
            Frac.append([Entropy_[j][i]/Latency[j][i] for i in range(len(Tau))])
        PLT_t = PLOT(Tau,Frac,D,X_L,Y_t,Name_EL_A3)
        
        
        f1 = Frac[0] 
        f2 = Frac[1]
        f3 = Frac[2]
        f4 = Frac[3]
        
        Frac[0]  = f3
        Frac[1]  = f4
        Frac[2]  = f2
        Frac[3]  = f1
        
        
        c1 = PLT_t.colors[0] 
        c2 = PLT_t.colors[1] 
        c3 = PLT_t.colors[2] 
        c4 = PLT_t.colors[3] 
        c5 = PLT_t.colors[4]
        PLT_t.colors[0] = c2
        PLT_t.colors[1] = c1
        PLT_t.colors[2] = c5
        PLT_t.colors[3] = c3
        #print(PLT_t.colors)
        
        
        p1 = PLT_t.Line_style[0] 
        p2 = PLT_t.Line_style[1] 
        p3 = PLT_t.Line_style[2] 
        p4 = PLT_t.Line_style[3] 
        p5 = PLT_t.Line_style[4] 
        PLT_t.Line_style[0] = p2
        PLT_t.Line_style[1] = p1
        PLT_t.Line_style[2] = p5
        PLT_t.Line_style[3] = p3
        
        
        PLT_t.scatter_line(True,270)

        
        





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
        
        ######################################################################################################
        ######################################################################################################
        ####################################FCP Approach3 Simulations##############################################
        #############################################################################################
        ###################################################################################
        #############################################################################################
        import json
        File_name = Name_File+'/FCPSim.json'
        with open(File_name , 'r') as file:
            # Serialize and save your data to the file
            data0 = json.loads(json.load(file))    
            
             
        
        
        Aa = data0['Alpha']
        
        Y_E = [data0['Entropy_Uniform'],data0['Entropy_Fair'],data0['Entropy_LARMIX']]
        
        D = ['Random','Single Location','Worst Case']
        ##################################Plots##################################################           
        
        Y_Label_E = 'Entropy (bits)'
        X_Label = r'Randomness $\tau$ ' 
        
        
        
        PLT = PLOT(Aa,Y_E,D,X_Label,Y_Label_E,Name_FCP_S3,True,False,False,True)
        PLT.Box_Plot(15,True)        



    
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
        
        ######################################################################################################
        ######################################################################################################
        ####################################FCP SC Simulations##############################################
        #############################################################################################
        ###################################################################################
        #############################################################################################
        import json
        File_name =Name_File+'/FCPSim.json'
        with open(File_name , 'r') as file:
        # Serialize and save your data to the file
            data0 = json.loads(json.load(file))    
        
         
        
        
        Aa = data0['Alpha']
        Aa = data0['Alpha']
        for i in range(len(Aa)):
            Aa[i] =1000*Aa[i]
            Y_E = [data0['Entropy_Uniform'],data0['Entropy_Fair'],data0['Entropy_LARMIX']]
        
        D = ['Random','Single Location','Worst Case']
        ##################################Plots##################################################           
        
        Y_Label_E = 'Entropy (bits)'
        X_Label = r'Radius $r$ ms' 
        
        
        
        PLT = PLOT(Aa,Y_E,D,X_Label,Y_Label_E,Name_FCP_S2,True,False,False,True)
        PLT.Box_Plot(15,True)
        
        
        


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
        
        ######################################################################################################
        ######################################################################################################
        ####################################FCP SC Simulations##############################################
        #############################################################################################
        ###################################################################################
        #############################################################################################
        import json
        File_name =Name_File+'/FCPSim.json'
        with open(File_name , 'r') as file:
        # Serialize and save your data to the file
            data0 = json.loads(json.load(file))    
        
         
        
        
        Aa = data0['Alpha']
        Aa = data0['Alpha']
        for i in range(len(Aa)):
            Aa[i] =1000*Aa[i]
            Y_E = [data0['Entropy_Uniform'],data0['Entropy_Fair'],data0['Entropy_LARMIX']]
        
        D = ['Random','Single Location','Worst Case']
        ##################################Plots##################################################           
        
        Y_Label_E = 'Entropy (bits)'
        X_Label = r'Radius $r$ ms' 
        
        
        
        PLT = PLOT(Aa,Y_E,D,X_Label,Y_Label_E,Name_FCP_S2,True,False,False,True)
        PLT.Box_Plot(15,True)
        
        
        






    
    def SC_Budget_FCP(self,Name_File,Iterations):
        #from Multi_Circles import CircularMixNet

        Class_SC = CircularMixNet_SC(self.num_targets,Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.H_N,self.N,self.rate,self.num_gateways,self.Percentile,Name_File) 


        #Sim.EL_Analysis('BaseLine'+str(W)+'Nodes_per_Layers',40,True)
        
        Budget = [5,10,15,20]
        
        for i in range(3):
            
        
            Class_SC.FCP_Budget(Iterations,Name_File+str(Budget[i+1]),Budget[i+1],True)     


        

        from Plot import PLOT
        Method = 'Greedy'
        
        X_L = r'Percentage of Corruption $C$'
        
        Y_f = "FCP"
        
        
        D = [r'LARMix $\tau=0.3$',r'LARMix $\tau=0.6$','Proportional',r'LARMix $\tau=0.9$','Uniform']
            
        B_FCP_A2 = 'Results' + '/' +'Fig_9a.png'
        
        X_values = [5,10,15,20]
        
        ##Fix the radius to be 30 ms sec################################
        
        ######################################################################################################
        ######################################################################################################
        import json
        File_name = Name_File + str(10)+'/FCP_Data.json'
        with open(File_name , 'r') as file:
            # Serialize and save your data to the file
            a = json.load(file)
            
        Keys = list(a['FCP'][Method].keys())
        #print(Keys)
        New_Data = {}
        for item in Keys:
            New_Data[item] = []
        
        Y_FCP = [[],[],[],[],[]]
        #print(Y_FCP)
        for i in range(len(Y_FCP)):
            #print(i,Y_FCP[i])
            next_x = a['FCP'][Method][Keys[i]][0]
            #print(next_x,type(next_x))
            Y_FCP[i].append( next_x)
        
            
        
        #print(Y_FCP)
        
        #print('ok')
        ######################################################################################################
        ######################################################################################################
        import json
        File_name = Name_File + str(15)+'/FCP_Data.json'
        with open(File_name , 'r') as file:
            # Serialize and save your data to the file
            a = json.load(file)
        
        for i in range(len(Y_FCP)):
            Y_FCP[i].append(a['FCP'][Method][Keys[i]][0])
        
        
        ######################################################################################################
        ######################################################################################################
        import json
        File_name = Name_File + str(20)+'/FCP_Data.json'
        with open(File_name , 'r') as file:
            # Serialize and save your data to the file
            a = json.load(file)
        
        for i in range(len(Y_FCP)):
            Y_FCP[i].append(a['FCP'][Method][Keys[i]][0])
        
        
        Y_B = [[],[],[],[],[]]
        
        
        for i,item in enumerate(Y_FCP):
            y_next = abs(polynomial_extrapolation([0.1,0.15,0.2],item, 0.05))
            Y_B[i] = [y_next]+item
            
            
        PLT_f = PLOT(X_values,Y_B,D,X_L,Y_f,B_FCP_A2)
        PLT_f.colors = [PLT_f.colors[4],PLT_f.colors[3],PLT_f.colors[2],PLT_f.colors[1],PLT_f.colors[0]]
        PLT_f.Line_style = [PLT_f.Line_style[4],PLT_f.Line_style[3],PLT_f.Line_style[2],PLT_f.Line_style[1],PLT_f.Line_style[0]]
        
        PLT_f.scatter_line(True,0.4)




    def MC_Budget_FCP(self,Name_File,Iterations):
        #from Multi_Circles import CircularMixNet

        Class_MC = CircularMixNet_MC(self.num_targets,Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.H_N,self.N,self.rate,self.num_gateways,self.Percentile,Name_File) 


        #Sim.EL_Analysis('BaseLine'+str(W)+'Nodes_per_Layers',40,True)
        
        Budget = [5,10,15,20]
        
        for i in range(3):
            
        
            Class_MC.FCP_Budget(Iterations,Name_File+str(Budget[i+1]),Budget[i+1],True)     


        

        Method = 'Greedy'
        
        from Plot import PLOT
        
        
        X_L = r'Percentage of Corruption $C$'
        
        Y_f = "FCP"
        
        
        D = [r'LARMix $\tau=0.3$',r'LARMix $\tau=0.6$','Proportional',r'LARMix $\tau=0.9$','Uniform']
            
        B_FCP_A1 = 'Results' + '/' +'Fig_9b.png'
        
        X_values = [5,10,15,20]
        
        ##Fix the radius to be 30 ms sec################################
        
        ######################################################################################################
        ######################################################################################################
        import json
        File_name = Name_File + str(10)+'/FCP_Data.json'
        with open(File_name , 'r') as file:
            # Serialize and save your data to the file
            a = json.load(file)
            
        #print(list(a['FCP'].keys()))
            
        Keys = list(a['FCP'][Method].keys())
        #print(Keys)
        New_Data = {}
        for item in Keys:
            New_Data[item] = []
        
        Y_FCP = [[],[],[],[],[]]
        #print(Y_FCP)
        for i in range(len(Y_FCP)):
            #print(i,Y_FCP[i])
            next_x = a['FCP'][Method][Keys[i]][0]
            #print(next_x,type(next_x))
            Y_FCP[i].append( next_x)
        
            
        
        #print(Y_FCP)
        
        
        ######################################################################################################
        ######################################################################################################
        import json
        File_name = Name_File + str(15)+'/FCP_Data.json'
        with open(File_name , 'r') as file:
            # Serialize and save your data to the file
            a = json.load(file)
        
        for i in range(len(Y_FCP)):
            Y_FCP[i].append(a['FCP'][Method][Keys[i]][0])
        
        
        ######################################################################################################
        ######################################################################################################
        import json
        File_name = Name_File + str(20)+'/FCP_Data.json'
        with open(File_name , 'r') as file:
            # Serialize and save your data to the file
            a = json.load(file)
            
        for i in range(len(Y_FCP)):
            Y_FCP[i].append(a['FCP'][Method][Keys[i]][0])
        
        #print(Y_FCP)
        
        
        Y_B = [[],[],[],[],[]]
        
        
        for i,item in enumerate(Y_FCP):
            y_next = abs(polynomial_extrapolation([0.1,0.15,0.2],item, 0.05))
            Y_B[i] = [y_next]+item
            
            
        PLT_f = PLOT(X_values,Y_B,D,X_L,Y_f,B_FCP_A1)
        PLT_f.colors = [PLT_f.colors[4],PLT_f.colors[3],PLT_f.colors[2],PLT_f.colors[1],PLT_f.colors[0]]
        PLT_f.Line_style = [PLT_f.Line_style[4],PLT_f.Line_style[3],PLT_f.Line_style[2],PLT_f.Line_style[1],PLT_f.Line_style[0]]
        
        PLT_f.scatter_line(True,0.4)




    def RM_Budget_FCP(self,Name_File,Iterations):
        #from Multi_Circles import CircularMixNet

        Class_RM =  Regional_MixNet(self.num_targets,Iterations,self.Capacity,self.run,self.delay1,self.delay2,self.H_N,self.N,self.rate,self.num_gateways,self.Percentile,Name_File) 


        #Sim.EL_Analysis('BaseLine'+str(W)+'Nodes_per_Layers',40,True)
        
        Budget = [5,17,25,30]
        
        for i in range(3):
            
        
            Class_RM.FCP_Budget(Iterations,Name_File+str(Budget[i+1]),Budget[i+1])     


        

        Method = 'Greedy'
        
        
        from Plot import PLOT
        
        
        X_L = r'Percentage of Corruption $C$'
        
        Y_f = "FCP"
        
        
        D = ['NA Mixnet + NA Clients','EU Mixnet + EU Clients','Global Mixnet + EU Clients','Global Mixnet + NA Clients']
            
        B_FCP_A3 = 'Results/Fig_9c.png'
        
        X_values = [5,10,15,20]
        ######################################################################################################
        ######################################################################################################
        import json
        File_name = Name_File+str(17)+'/FCP_Data.json'
        
        with open(File_name , 'r') as file:
            # Serialize and save your data to the file
            a = json.load(file)
            
         
        
        
        
        Y_FCP = []
        for item in a['FCP'][Method]:
            Y_FCP.append(a['FCP'][Method][item])
            
        y1 = Y_FCP[0]
        y2= Y_FCP[1]
        y3 = Y_FCP[2]
        y4 = Y_FCP[3]   
        
        Y_FCP[0] = y4
        Y_FCP[1] = y3
        Y_FCP[2] = y1
        Y_FCP[3] = y2
        
        Y_FCP_ = [[],[],[],[]]
        #print(Y_FCP)
        for i in range(len(Y_FCP)):
            #print(i,Y_FCP[i])
            next_x = Y_FCP[i][0]
            #print(next_x,type(next_x))
            Y_FCP_[i].append( next_x)
        
        
        ##Fix the yau to be 0.6################################
        
        
        ######################################################################################################
        ######################################################################################################
        import json
        File_name = Name_File+str(25)+'/FCP_Data.json'
        
        with open(File_name , 'r') as file:
            # Serialize and save your data to the file
            a = json.load(file)
            
         
        
        
        
        Y_FCP = []
        for item in a['FCP'][Method]:
            Y_FCP.append(a['FCP'][Method][item])
            
        y1 = Y_FCP[0]
        y2= Y_FCP[1]
        y3 = Y_FCP[2]
        y4 = Y_FCP[3]   
        
        Y_FCP[0] = y4
        Y_FCP[1] = y3
        Y_FCP[2] = y1
        Y_FCP[3] = y2
        
        
        for i in range(len(Y_FCP)):
            #print(i,Y_FCP[i])
            next_x = Y_FCP[i][0]
            #print(next_x,type(next_x))
            Y_FCP_[i].append( next_x)
        
        
        #print(Y_FCP_)
        
        #
        
        
        
        
        ######################################################################################################
        ######################################################################################################
        import json
        File_name = Name_File+str(30)+'/FCP_Data.json'
        
        with open(File_name , 'r') as file:
            # Serialize and save your data to the file
            a = json.load(file)
            
         
        
        
        
        Y_FCP = []
        for item in a['FCP'][Method]:
            Y_FCP.append(a['FCP'][Method][item])
            
        y1 = Y_FCP[0]
        y2= Y_FCP[1]
        y3 = Y_FCP[2]
        y4 = Y_FCP[3]   
        
        Y_FCP[0] = y4
        Y_FCP[1] = y3
        Y_FCP[2] = y1
        Y_FCP[3] = y2
        
        
        for i in range(len(Y_FCP)):
            #print(i,Y_FCP[i])
            next_x = Y_FCP[i][0]
            #print(next_x,type(next_x))
            Y_FCP_[i].append( next_x)
        
        
        
        Y_B = [[],[],[],[]]
        
        
        for i,item in enumerate(Y_FCP_):
            y_next = abs(polynomial_extrapolation([0.1,0.15,0.2],item, 0.05))
            Y_B[i] = [y_next]+item
        
        
        PLT_f = PLOT(X_values,Y_B,D,X_L,Y_f,B_FCP_A3)
        PLT_f.colors = [PLT_f.colors[2],PLT_f.colors[1],PLT_f.colors[0],PLT_f.colors[4],PLT_f.colors[0]]
        PLT_f.Line_style = [PLT_f.Line_style[2],PLT_f.Line_style[1],PLT_f.Line_style[0],PLT_f.Line_style[4],PLT_f.Line_style[0]]
        
        PLT_f.scatter_line(True,0.4)
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        

