# -*- coding: utf-8 -*-
"""
Some ToyExamples: tests

In this file we provide some examples of running the functions exists in the file  of different routing 

mechanisms, namely, Regional.py, Multi_Circles.py and Single_Circle.py.
"""

from Multi_Circles import CircularMixNet_MC
from Single_Circle import CircularMixNet_SC
from Regional import Regional_MixNet
import numpy as np
#General paramters
#The following parameters are considered throughout the paper for execution, for sure you can play around and use others, 
#but note that H_N and rate are some factors to increase number of generated massages in the simulations and specifically 
#run indicated the amount of time you want the simulations environment simulate a mix-net. Capacity is the capacity of 
#the mix-nodes and Percentile indicates the measure of latency which can be baes on median or 95 percentile

W = 60
N = 3*W
num_gateways = W
delay1 = 0.05
delay2 = 0.0001/2
Capacity = 10000000000000000000000000000000000000000000000000000000000000000
H_N = round(N/3)
rate = 100
num_targets = 200
Iterations = 20
run = 0.5
Percentile = [50,95]    

################################################################################################################################
##########################Single Circles########################################################################################
################################################################################################################################
#Please uncomments any of the following commented out function to test the corresponding function with your input.


Name_File = 'Single_Circle_Test'

Class_SC = CircularMixNet_SC(num_targets,Iterations,Capacity,run,delay1,delay2,H_N,N,rate,num_gateways,Percentile,Name_File) 

"""
1st: #To make a mixnet configuration with the data of nodes' latency

"""

#Class_SC.MixNet_Creation()


"""
2nd: #Applies LARMix routing for a list of latency List for value of \tau = Tau

"""


# Class_SC.LARMIX(List,Tau)

"""
3rd: The following function initializes instances of mix networks based on the number of iterations specified. 
#It applies the Single Circle Routing mechanism alongside the within-circle routing using the value of \tau to configure 
#the initial setup and generate the data required for analyzing the Single Circle approach.

"""

# Class_SC.PreProcessing(Iterations)



"""
4th: This function generates the basic analysis of the Single Circle approach, specifically evaluating latency and entropy 
#as described in Figure 5.

"""


#Class_SC.EL_Analysis('BaseLine_SC', Iterations)
"""
5th: This function provides results for the trade-off between the amount of randomness introduced by the routing approaches 
# and the mixing delays. The value 0.2 represents the maximum allowed end-to-end delay, 
#and 'LARMIX' refers to the within-circle routing approach.

"""


#Class_SC.E2E(0.2, Iterations, 'LARMIX', 'E2E_SC')


"""
6th: This function assesses the results of the Single Circle strategy when varying the alpha parameter.

"""


#Class_SC.EL_Analysis_alpha('Alpha_SC', Iterations)


"""
7th: This function evaluates the behavior of the Single Circle approach while considering adversarial 
#mix nodes attempting to corrupt some mix nodes in the network.

"""


#Class_SC.FCP(Iterations, 'FCP_SC')


"""
8th: This function builds on the previous function, changing the value of the corrupted nodes allocated to the adversary 
# to measure the impact of its budget. The list includes some predefined budget values for the adversary.
"""


#Class_SC.FCP_Budget(Iterations, 'FCP_Budget_SC', [0.1, 0.2, 0.3, 0.4])




################################################################################################################################
##########################Multiple Circles######################################################################################
################################################################################################################################
#Please uncomments any of the following commented out function to test the corresponding function with your input.


Name_File = 'Multiple_Circles_Test'

Class_MC = CircularMixNet_MC(num_targets,Iterations,Capacity,run,delay1,delay2,H_N,N,rate,num_gateways,Percentile,Name_File) 

"""
1st: #To make a mixnet configuration with the data of nodes' latency

"""

#Class_MC.MixNet_Creation()


"""
2nd: #Applies LARMix routing for a list of latency List for value of \tau = Tau

"""


# Class_MC.LARMIX(List,Tau)

"""
3rd: The following function initializes instances of mix networks based on the number of iterations specified. 
#It applies the Multiple Circles Routing mechanism alongside the within-circle routing using the value of \tau to configure 
#the initial setup and generate the data required for analyzing the Multiple Circles approach.

"""

# Class_MC.PreProcessing(Iterations)



"""
4th: This function generates the basic analysis of the Multiple Circles approach, specifically evaluating latency and entropy 
#as described in Figure 5.

"""


#Class_MC.EL_Analysis('BaseLineMSC', Iterations)

"""
5th: This function provides results for the trade-off between the amount of randomness introduced by the routing approaches 
# and the mixing delays. The value 0.2 represents the maximum allowed end-to-end delay, 
#and 'LARMIX' refers to the within-circle routing approach.

"""


#Class_MC.E2E(0.2, Iterations, 'LARMIX', 'E2E_MC')


"""
6th: This function assesses the results of the Multiple Circles strategy when varying the alpha parameter.

"""


#Class_MC.EL_Analysis_alpha('Alpha_MC', Iterations)


"""
7th: This function evaluates the behavior of the Multiple Circles approach while considering adversarial 
#mix nodes attempting to corrupt some mix nodes in the network.

"""


#Class_MC.FCP(Iterations, 'FCP_MC')


"""
8th: This function builds on the previous function, changing the value of the corrupted nodes allocated to the adversary 
# to measure the impact of its budget. The list includes some predefined budget values for the adversary.
"""


#Class_MC.FCP_Budget(Iterations, 'FCP_Budget_MC', [0.1, 0.2, 0.3, 0.4])





################################################################################################################################
##########################Regional Mixnets######################################################################################
################################################################################################################################
#Please uncomments any of the following commented out function to test the corresponding function with your input.


Name_File = 'Regional_Mixnet_Test'

Class_RM = Regional_MixNet(num_targets,Iterations,Capacity,run,delay1,delay2,H_N,N,rate,num_gateways,Percentile,Name_File) 

"""
1st: This class divides the dataset into clusters or regions to emulate different regions.
"""

#Class_RM.Clustering()



"""
2nd: To create a MixNet configuration using the nodes' latency data.
The first argument specifies the region (e.g., 'EU', 'NA', 'Global + EU', or 'Global + NA'), 
and the second argument indicates the number of nodes at each layer.
"""


#Class_RM.MixNet_Creation('EU', 10)





"""
4th: This function generates the basic analysis of the Multiple Circles approach, 
specifically evaluating latency and entropy as described in Figure 5. 
'K_Cluster' represents the number of clusters.
"""



#Class_RM.EL_Analysis1(Name_, Iteration, K_Cluster)






"""
5th: This function provides results for the trade-off between the amount of randomness introduced 
by the routing approaches and the mixing delays. The value 0.2 represents the maximum allowed end-to-end delay.
"""




#Class_RM.E2E(0.2, Iterations, 'E2E_RM')




"""
7th: This function evaluates the behavior of the Regional MixNets approach while considering 
adversarial mix nodes attempting to corrupt some mix nodes in the network.
"""




#Class_RM.FCP(Iterations, 'FCP_RM')



"""
8th: This function builds on the previous function, modifying the value of corrupted nodes allocated 
to the adversary to measure the impact of its budget. The list contains predefined budget values for the adversary.
"""




#Class_MC.FCP_Budget(Iterations, 'FCP_Budget_RM', [0.2, 0.3, 0.4, 0.5])























