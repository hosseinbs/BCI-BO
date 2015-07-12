import numpy as np
import matplotlib.pyplot as plt
import math
from pylab  import figure
from my_plotter import *


if __name__ == '__main__':
    
    
    
    BP_y = [(72.96,78.62,78.62,76.11,79.25,79.88), (64.45,65.38,65.75,65.00,67.04,66.67), (69.45,71.86,74.26,72.04,69.75,72.6)]
    labels = ['BP_O3','BP_S4','BP_X11']

    plotter( BP_y, 49, 85, 'BP', labels)
    
    
    logBP_y = [(74.22,79.25,79.25,77.36,81.77,81.77), (62.23,66.49,66.30,65.38,66.86,66.86), (69.82,72.97,73.15,71.86,74.63,74.63)]
    labels = ['LOGBP_O3','LOGBP_S4','LOGBP_X11']

    plotter( logBP_y, 49, 85, 'logBP', labels)
    
    
    wackermann_y = [(56.61,57.24,58.24,54.72,54.72,59.75), (57.97,57.6,59.82,55.75,57.97,58.71), (60,50,57.24,61.49,60.56,62.23)]
    labels = ['wackerman_O3','wackerman_S4','wackerman_X11']

    plotter( wackermann_y, 49, 85, 'wackerman', labels)
    
    plt.show()

#    y_RF = [(77.98,76.72,76.72,79.87), (70.74,74.44,80.92,75.18),(75.92,73.51,77.03,78.33),(76.11,77.36,58.5, 54.72), (65,65.38,53.34,55.75), (72.04,71.86,60,61.49)]
#    labels = ['BO_RF_O3','BO_RF_S4','BO_RF_X11','RF_g_search_O3','RF_g_search_S4','RF_g_search_X11']
#    BO_plotter( y_RF, 49, 84, 'BO_RF_2', labels)
    
#    plt.show()
    
    
    ##bp + logbp BO + wackerman
#    y_RF = [(81.1320754717,84.2767295597,82.3899371069),(73.5185185185,77.962962963,79.6296296296),(79.6296296296,76.4814814815,77.2222222222),(77.9874213836,79.2452830189, 76.7295597484), 
#            (77.2222222222,77.2222222222,73.8888888889), (71.1111111111,76.1111111111,72.5925925926), (76.11,77.36,54.72), (65,65.38,55.75), (72.04,71.86,61.49)]
#    labels = ['BO_RF_avg_O3','BO_RF_avg_S4','BO_RF_avg_X11', 'BO_RF_min_O3','BO_RF_min_S4','BO_RF_min_X11' ,'RF_g_search_O3','RF_g_search_S4','RF_g_search_X11']
#    
#    BO_plotter( y_RF, 52, 84, 'BO_RF_1', labels)
#    
#    plt.show()


#
#    y_RF = [ (70.74,74.44,80.92,75.18), (65,65.38,53.34,55.75)]
#    labels = ['Bayesian_Optimization+Random_Forest','grid_search+Random_Forest']
#    BO_plotter( y_RF, 49, 84, 'BO_RF_2', labels)
#    
#    plt.show()
    
    
    ##bp + logbp BO + wackerman
#    y_RF = [(81.1320754717,84.2767295597,82.3899371069),(77.9874213836,79.2452830189, 76.7295597484), (76.11,77.36,54.72)]
#    labels = ['Bayesian_Optimization+Random_Forest+average', 'Bayesian_Optimization+Random_Forest+minimum','grid_search+Random_Forest']
##    
#    BO_plotter( y_RF, 52, 84, 'BO_RF_1', labels)
##    
#    plt.show()
    