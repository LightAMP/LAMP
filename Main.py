"""
Main File: In this file we evaluate LAMP
"""

            
from LAMP import LAMP
C = LAMP()
#To evaluate LAMP, the first step is to configure a mixed network using the following function.
# Note that the number of iterations used for this configuration should not be exceeded in other experiments. 
#For example, if the iteration count is set to 10 in the initial configuration, 
#subsequent experiments should also use a maximum of 10 iterations.
C.data_initialization(50)


#Fig 5a, 5d.
#The function takes two arguments: the first argument is used to specify the name of the experiment, 
#the second argument represents the number of iterations 

C.SC_Baselines('SC_BaseLine',5)


#Fig 5b, 5e.
#The function takes two arguments: the first argument is used to specify the name of the experiment, 
#the second argument represents the number of iterations 
C.MC_Baselines('MC_BaseLine',2)


#Fig 5c, 5f.
#The function takes two arguments: the first argument is used to specify the name of the experiment, 
#the second argument represents the number of iterations 
C.RM_Baselines('RM_BaseLine',5)



#Fig 6a, 6b. Fig 8c
#The function takes two arguments: the first argument is used to specify the name of the experiment, 
#the second argument represents the number of iterations 

C.RM_Simulations_FCP('RM_Simulations_and_FCP',10)



#Fig 8a
#The function takes two arguments: the first argument is used to specify the name of the experiment, 
#the second argument represents the number of iterations 

C.SC_Simulation_FCP('SC_Simulation_and_FCP',50)


#Fig 8b
#The function takes two arguments: the first argument is used to specify the name of the experiment, 
#the second argument represents the number of iterations 

C.MC_Simulation_FCP('MC_Simulation_and_FCP',50)























