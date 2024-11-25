from LAMP import LAMP
import os


class Main_functions(object):
    
    def __init__(self,Fun,State = False):
        
        #assert Iterations <201, 'The maximum number of iterations should not exceed 200'
    
        self.Fun = Fun
        self.It = 25
        

        if not ( os.path.exists("LARMIX__2.json") and os.path.exists("LARMIX__2.json") and os.path.exists("LARMIX.json") ):
            
            

            C = LAMP()
            #To evaluate LAMP, the first step is to configure a mixed network using the following function.
            # Note that the number of iterations used for this configuration should not be exceeded in other experiments. 
            #For example, if the iteration count is set to 10 in the initial configuration, 
            #subsequent experiments should also use a maximum of 10 iterations.
            
            C.data_initialization(200)
        if not Fun ==0:
            self.Execution()
        else:
            self.Execution_()
            
    def Execution2(self):
        
        if self.Fun == 'Fig_5_SC':
            #Fig 5a, 5d, 5g.
            #The function takes two arguments: the first argument is used to specify the name of the experiment, 
            #the second argument represents the number of iterations 
            C = LAMP()
            C.SC_Baselines('SC_BaseLine',self.It)
        elif self.Fun == 'Fig_5_MC':
            #Fig 5b, 5e, 5e.
            #The function takes two arguments: the first argument is used to specify the name of the experiment, 
            #the second argument represents the number of iterations 
            C = LAMP()
            C.MC_Baselines('MC_BaseLine',self.It)
        elif self.Fun == 'Fig_5_RM':
            #Fig 5c, 5f, 5i.
            #The function takes two arguments: the first argument is used to specify the name of the experiment, 
            #the second argument represents the number of iterations 
            C = LAMP()
            C.RM_Baselines('RM_BaseLine',self.It)  
        elif self.Fun == 'Fig_6_7_8_RM':
            
            #Fig 6a, 6b, Fig 7c, Fig 8c
            #The function takes two arguments: the first argument is used to specify the name of the experiment, 
            #the second argument represents the number of iterations 
            C = LAMP()
            C.RM_Simulations_FCP('RM_Simulations_and_FCP',self.It)
        elif self.Fun == 'Fig_7_8_SC':
            #Fig 7a,Fig 8a.
            #The function takes two arguments: the first argument is used to specify the name of the experiment, 
            #the second argument represents the number of iterations 
            C = LAMP()
            C.SC_Simulation_FCP('SC_Simulation_and_FCP',self.It)
        elif self.Fun == 'Fig_7_8_MC':        
            #Fig 7b, Fig 8b.
            #The function takes two arguments: the first argument is used to specify the name of the experiment, 
            #the second argument represents the number of iterations 
            C = LAMP()
            C.MC_Simulation_FCP('MC_Simulation_and_FCP',self.It)                        

        elif self.Fun == 'Fig_9_SC':
            #Fig 9a.
            #The function takes two arguments: the first argument is used to specify the name of the experiment, 
            #the second argument represents the number of iterations 
            C = LAMP()
            C.SC_Budget_FCP('SC_FCP_Budget',self.It)
        elif self.Fun == 'Fig_9_MC':        
            #Fig 9b.
            #The function takes two arguments: the first argument is used to specify the name of the experiment, 
            #the second argument represents the number of iterations 
            C = LAMP()
            C.MC_Budget_FCP('MC_FCP_Budget',self.It)  
        elif self.Fun == 'Fig_9_RM':
            
            #Fig 9c.
            #The function takes two arguments: the first argument is used to specify the name of the experiment, 
            #the second argument represents the number of iterations 
            C = LAMP()
            C.RM_Budget_FCP('RM_FCP_Budget',self.It)
            
    def Execution_(self):
        #All the experiments
        C = LAMP()

        
        #Fig 5a, 5d, 5g.
        #The function takes two arguments: the first argument is used to specify the name of the experiment, 
        #the second argument represents the number of iterations 
        
        C.SC_Baselines('SC_BaseLine',5)
        
        
        
        #Fig 5b, 5e, 5e.
        #The function takes two arguments: the first argument is used to specify the name of the experiment, 
        #the second argument represents the number of iterations 
        C.MC_Baselines('MC_BaseLine',5)
        
        
        
        #Fig 5c, 5f, 5i.
        #The function takes two arguments: the first argument is used to specify the name of the experiment, 
        #the second argument represents the number of iterations 
        C.RM_Baselines('RM_BaseLine',5)

        #Fig 6a, 6b, Fig 7c, Fig 8c
        #The function takes two arguments: the first argument is used to specify the name of the experiment, 
        #the second argument represents the number of iterations 
        
        C.RM_Simulations_FCP('RM_Simulations_and_FCP',10)

        
        #Fig 7a,Fig 8a.
        #The function takes two arguments: the first argument is used to specify the name of the experiment, 
        #the second argument represents the number of iterations 
        
        C.SC_Simulation_FCP('SC_Simulation_and_FCP',25)
        

        
        #Fig 7b, Fig 8b.
        #The function takes two arguments: the first argument is used to specify the name of the experiment, 
        #the second argument represents the number of iterations 
        
        C.MC_Simulation_FCP('MC_Simulation_and_FCP',25)
        

        
        #Fig 9a
        #The function takes two arguments: the first argument is used to specify the name of the experiment, 
        #the second argument represents the number of iterations 
        
        C.SC_Budget_FCP('SC_FCP_Budget',25)
        
        
        
        
        #Fig 9b
        #The function takes two arguments: the first argument is used to specify the name of the experiment, 
        #the second argument represents the number of iterations 
        
        C.MC_Budget_FCP('MC_FCP_Budget',25)
        

        #Fig 9c
        #The function takes two arguments: the first argument is used to specify the name of the experiment, 
        #the second argument represents the number of iterations 
        
        C.RM_Budget_FCP('RM_FCP_Budget',25)



    def Execution(self):
        
        if self.Fun == 12 or self.Fun == 5 :
            #Fig5. E1 and E2
            #The function takes two arguments: the first argument is used to specify the name of the experiment, 
            #the second argument represents the number of iterations 
            C = LAMP()
            C.SC_Baselines('SC_BaseLine',5)

            C.MC_Baselines('MC_BaseLine',5)

            C.RM_Baselines('RM_BaseLine',5)  
            
        elif self.Fun == 3 or self.Fun == 6:
            #Fig6 and E3
            C = LAMP()
            C.RM_Simulations_FCP('RM_Simulations_and_FCP',10)
            
        elif self.Fun == 4 or self.Fun ==7 or self.Fun ==8 :
            #Fig7, Fig8 and E4
            #The function takes two arguments: the first argument is used to specify the name of the experiment, 
            #the second argument represents the number of iterations 
            C = LAMP()
            C.RM_Simulations_FCP('RM_Simulations_and_FCP',10)
            C.SC_Simulation_FCP('SC_Simulation_and_FCP',25)
            C.MC_Simulation_FCP('MC_Simulation_and_FCP',25) 
            
                      

        elif self.Fun == 9:
            #Fig9
            #The function takes two arguments: the first argument is used to specify the name of the experiment, 
            #the second argument represents the number of iterations 
            C = LAMP()
            C.SC_Budget_FCP('SC_FCP_Budget',25)

            C.MC_Budget_FCP('MC_FCP_Budget',25)  
  
            C.RM_Budget_FCP('RM_FCP_Budget',25)
            



































