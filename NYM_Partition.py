# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:56:51 2023

@author: Mahdi
"""



import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

def Latency_reader(a):
    b = ''
    for char in a:            
        if char=='m':
            break
        b = b +char
        if char =='u' or char=='n':
            b = '1'
            break
    return float(b)
    


    
    
def Box_Plot_Extraction(A,B,C=True):
    I = 0
    
   
    Latency_A = []
    Latency_B = []
    
    
    for i in range(len(A)):
        ID1 = i
        for j in range(len(A)):
            ID2 = j   
            I_key = A[ID2]['i_key']
            In_Latency = A[ID1]['latency_measurements'][str(I_key)]
            if type(In_Latency) == float:
                
                delay_distance = float(In_Latency)
            else:
                delay_distance =  Latency_reader(In_Latency)
            if int(delay_distance) == 0 or delay_distance<0 :
                delay_distance =1                       
            Latency_A.append(delay_distance/2000)
            
    
    
    for i in range(len(A)):
        ID1 = i
        for j in range(len(B)):
            ID2 = j   
            I_key = B[ID2]['i_key']
            In_Latency = A[ID1]['latency_measurements'][str(I_key)]
            if type(In_Latency) == float:
                
                delay_distance = float(In_Latency)
            else:
                delay_distance =  Latency_reader(In_Latency)
            if int(delay_distance) == 0  or delay_distance<0:
                delay_distance =1                       
            Latency_B.append(delay_distance/2000) 
    if C:
        List = [(Latency_A),(Latency_B)]
    else:
        List = [Latency_A,Latency_B]

    return List
def classify_and_plot(data_list):
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

    # Add legend
    plt.legend()

    # Save the image

    plt.savefig('D:/Approach3/images/NYM_New_globe_map117.png',format='png', dpi=600)

    # Show the plot
    plt.show()
    return europe, asia, north_america, south_america, africa, australia


def Plot_Box(LIST, D, Name):
    # Labels for the x-axis
    x_labels = D

    # Create a box plot
    box_plot = plt.boxplot(LIST, labels=x_labels, patch_artist=True)

    # Set the color for the boxplot
    colors = ['blue', 'red']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)

    # Add labels and title
    XL = 'Region'
    YL = 'Latency (Sec)'

    if True:
        plt.grid(linestyle='--')

    # Set the y-axis limit to 0.2
    plt.ylim(0, 0.122)

    plt.xlabel(XL)
    plt.ylabel(YL)

    plt.tight_layout()
    plt.savefig(Name, format='png', dpi=600)
    plt.show()

# Example usage:
# Plot_Box([np.random.rand(10), np.random.rand(10)], ['A', 'B'], 'box_plot.png')








def Distinguish(A,B):
    
    A_ = []
    B_ = []
    C_ = []
    
    for i in range(len(A)):
        ID1 = i
        for j in range(len(A)):
            Latency_A = 0
            ID2 = j   
            I_key = A[ID2]['i_key']
            In_Latency = A[ID1]['latency_measurements'][str(I_key)]
            
            delay_distance = float(In_Latency)
            if int(delay_distance) == 0 or delay_distance<0 :
                delay_distance =1                       
            Latency_A = Latency_A+(delay_distance/2000)
        A_.append([Latency_A/len(A),i])
            
    for i in range(len(A)):
        ID1 = i
        for j in range(len(B)):
            Latency_B = 0
            ID2 = j   
            I_key = B[ID2]['i_key']
            In_Latency = A[ID1]['latency_measurements'][str(I_key)]
            
            delay_distance = float(In_Latency)
            if int(delay_distance) == 0 or delay_distance<0 :
                delay_distance =1                       
            Latency_B = Latency_B+(delay_distance/2000)
        B_.append([Latency_B/len(B),i]) 
        
    
    for k in range(len(A_)):
        
        if A_[k][0]>B_[k][0]:
            C_.append(k)
                

    return C_









def find_central_point(data):
    # Check if the input data is not empty
    if not data:
        return None

    # Calculate average longitude and latitude
    total_longitude = 0
    total_latitude = 0
    num_points = len(data)

    for point in data:
        #print(float(point['longitude']),type(float(point['longitude'])))
        total_longitude += float(point['longitude'])
        total_latitude += float(point['latitude'])

    average_longitude = total_longitude / num_points
    average_latitude = total_latitude / num_points

    # Find the dictionary with the closest longitude and latitude to the averages
    central_point = min(data, key=lambda point: abs(float(point['longitude']) - average_longitude) + abs(float(point['latitude']) - average_latitude))

    return central_point 





#initial_centers = np.array([[2, 2], [8, 8]])


def centers(data):
    CENTERS = {}
    
    Names = ['europe', 'asia', 'north_america', 'south_america', 'africa','australia']
    for Name in Names:
        CENTERS[Name] = {}
    
    for continent in Names:
        
        C_ = find_central_point(data[continent])
        i = 0
        for item in data['Global']:
            if C_ == item:
                I = i
                break
            i = i+1
        CENTERS[continent] = I
        
    return CENTERS
            
                
        
    
    
    





# Example usage:
import json

with open('D:/Approach3/117_nodes_latency_December_2023_cleaned_up_9_no_intersection_1.json') as json_file: 

    data_list = json.load(json_file) # Your list of dictionaries

europe, asia, north_america, south_america, africa, australia = classify_and_plot(data_list)




print(len(europe)+len(asia)+ len(north_america)+ len(south_america)+ len(africa)+ len(australia))



print(data_list[0])





import json

with open('250_nodes_latency_December_2023_cleaned_up_2.json') as json_file: 

    data0 = json.load(json_file) # Your list of dictionaries

##########################Center of Clusters NYM dataset##################################################

Names = ['europe', 'asia', 'north_america', 'south_america', 'africa','australia']
Data_ = {}
for item in Names:
    Data_[item] = eval(item)

Data_['Global'] = data_list



'''

Centers = centers(Data_)




print(Centers)

'''


#############NYM DATASET ANALYSIS###########################

EU = europe


NA = north_america



AS = asia


print(len(africa),len(australia),len(south_america),len(asia))

D = ['NA','AS','EU']

DD = {}
for item in D:
    DD[item] = eval(item)

Test_Data = {'Mean':{},'Variance':{}}

for item in D:
    for term in D:
        if item == term:
            pass
        else:
            
            List = Box_Plot_Extraction(DD[item],DD[term],False)
          
            Plot_Box(List,[item,term],'D:/Approach3/Inter_Polated_NYM_250/' +item+term+'.png')




