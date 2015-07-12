import numpy as np
import matplotlib.pyplot as plt
import math
from pylab  import figure
from my_plotter import *
import os
import sys
sys.path.append('./BCI_Framework')

import Main
import Single_Job_runner as SJR

import os
import re


if __name__ == '__main__':
    
    bciciv1 = Main.Main('BCI_Framework','BCICIII3b','RANDOM_FOREST', 'BP', 'ALL', -1, 'python')
    res_path = bciciv1.config.configuration['results_opt_path_str']
    classifiers_dict = {'Boosting':0, 'LogisticRegression':1, 'RANDOM_FOREST':2,'SVM':3, 'LDA':4, 'QDA':5 , 'MLP':6}
    features_dict = {'BP':0, 'logbp':1, 'wackerman':2, 'morlet':3, 'AR':4}
    results = np.zeros((len(classifiers_dict),len(features_dict), bciciv1.config.configuration["number_of_subjects"]))
    discarded_periods = np.empty((len(classifiers_dict),len(features_dict), bciciv1.config.configuration["number_of_subjects"]), dtype='S10')
    subjects_dict = {}
    
    for ind, subj in enumerate(bciciv1.config.configuration["subject_names_str"]):
        subjects_dict.update({subj:ind})
        
    for dirname, dirnames, filenames in os.walk(res_path):
#        for subdirname in dirnames:
#            fold_name = os.path.join(dirname, subdirname)
#            print fold_name
        for filename in filenames:
#            slash_indices = re.search('0', filename)
            if filename[-4:] != '.pkl':
                file_name = os.path.join(dirname, filename)
                backslash_indices = [m.start() for m in re.finditer("\\\\", file_name)]
                underline_indices = [m.start() for m in re.finditer("_", file_name)]
                
                feature_ext_name = file_name[backslash_indices[-2]+1:backslash_indices[-1]]
                classifier_name = file_name[backslash_indices[-3]+1:backslash_indices[-2]]
                subj = file_name[underline_indices[-1]+1:-4]
    #            print feature_ext_name, classifier_name, subj
                npzfile = np.load(file_name)
                error = npzfile['error']
                accuracy = 100 - error*100
                results[classifiers_dict[classifier_name], features_dict[feature_ext_name],subjects_dict[subj]] = accuracy
                discarded_periods[classifiers_dict[classifier_name], features_dict[feature_ext_name],subjects_dict[subj]] = file_name[backslash_indices[-1]+1:underline_indices[2]]
                  
#            with open(file_name,'r') as my_file:
#                
#                error = float(my_file.readline())
#                accuracy = 100 - error*100
#                results[classifiers_dict[classifier_name], features_dict[feature_ext_name],subjects_dict[subj]] = accuracy
##                print file_name[backslash_indices[-1]+1:underline_indices[1]]
#                discarded_periods[classifiers_dict[classifier_name], features_dict[feature_ext_name],subjects_dict[subj]] = file_name[backslash_indices[-1]+1:underline_indices[2]]
#            
#            print backslash_indices

    for feature in features_dict.keys():
        f_ind = features_dict[feature]
        feature_ext_y = []
        labels = []
        for subject in subjects_dict.keys():
            subj_ind = subjects_dict[subject]
            
            feature_ext_y.append(tuple(results[:,f_ind,subj_ind]))
            labels.append(feature + '_' + subject)
            
#        plotter( feature_ext_y, math.floor(np.min(feature_ext_y) - 1), math.floor(np.max(feature_ext_y) + 1), feature, labels)
        plotter( feature_ext_y, 46, 97, feature, labels)    
        
    for subject in subjects_dict.keys():
        for feature in features_dict.keys():
            print subject, feature, discarded_periods[:, features_dict[feature],subjects_dict[subject]]
#    BP_y = [(72.96,78.62,78.62,76.11,79.25,79.88), (64.45,65.38,65.75,65.00,67.04,66.67), (69.45,71.86,74.26,72.04,69.75,72.6)]
#    labels = ['BP_O3','BP_S4','BP_X11']

#    plotter( BP_y, 64, 81, 'BP', labels)
    
    
#    logBP_y = [(74.22,79.25,79.25,77.36,81.77,81.77), (62.23,66.49,66.30,65.38,66.86,66.86), (69.82,72.97,73.15,71.86,74.63,74.63)]
#    labels = ['LOGBP_O3','LOGBP_S4','LOGBP_X11']

#    plotter( logBP_y, 61, 84, 'logBP', labels)
    
    
#    wackermann_y = [(56.61,57.24,58.24,54.72,54.72,59.75), (57.97,57.6,59.82,55.75,57.97,58.71), (60,50,57.24,61.49,60.56,62.23)]
#    labels = ['wackerman_O3','wackerman_S4','wackerman_X11']

#    plotter( wackermann_y, 49, 65, 'wackerman', labels)
    


#    y_RF = [(77.98,76.72,76.72,79.87), (70.74,74.44,80.92,75.18),(75.92,73.51,77.03,78.33),(76.11,77.36,58.5, 54.72), (65,65.38,53.34,55.75), (72.04,71.86,60,61.49)]
#    labels = ['BO_RF_O3','BO_RF_S4','BO_RF_X11','RF_grid_search_O3','RF_grid_search_S4','RF_grid_search_X11']
#    BO_plotter( y_RF, 49, 83, 'BO_RF', labels)
    
    plt.show()