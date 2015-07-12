import numpy as np
import math
import os
import sys
from sklearn.metrics import precision_recall_fscore_support

sys.path.append('./BCI_Framework')

import Main
import Single_Job_runner as SJR
import my_plotter
import os
import re
import Configuration_BCI
import matplotlib

def read_optimal_accuracies(dataset_name, CSPorALL):
    
    bciciv1 = Main.Main('BCI_Framework', dataset_name,'RANDOM_FOREST', 'BP', CSPorALL, -1, 'python')
    res_path = bciciv1.config.configuration['results_opt_path_str']
    test_data_path = bciciv1.config.configuration['test_data_dir_name_str']
    classifiers_dict = {'Boosting':0, 'LogisticRegression':1, 'RANDOM_FOREST':2,'SVM':3, 'LDA':4, 'QDA':5 , 'MLP':6}
    features_dict = {'BP':0, 'logbp':1, 'morlet':2}
    results = np.zeros((len(classifiers_dict),len(features_dict), bciciv1.config.configuration["number_of_subjects"]))
    statistical_test_results = np.zeros((len(classifiers_dict) * len(features_dict), bciciv1.config.configuration["number_of_subjects"]))
    statistical_test_results_titles = [None] * (len(classifiers_dict) * len(features_dict))
    discarded_periods = np.empty((len(classifiers_dict),len(features_dict), bciciv1.config.configuration["number_of_subjects"]), dtype='S10')
    subjects_dict = {}
    subjects_true_labels_dict = {}
    
    
    for ind, subj in enumerate(bciciv1.config.configuration["subject_names_str"]):
        subjects_dict.update({subj:ind})
        if dataset_name == 'SM2' or dataset_name == 'BCICIV1':
            subjects_true_labels_dict[subj] = np.loadtxt(os.path.join(test_data_path, subj + '_Y.txt'))

        
    for dirname, dirnames, filenames in os.walk(res_path):
        for filename in filenames:

            if filename[-4:] != '.pkl':
                file_name = os.path.join(dirname, filename)
                backslash_indices = [m.start() for m in re.finditer("\\\\", file_name)]
                underline_indices = [m.start() for m in re.finditer("_", file_name)]
                
                feature_ext_name = file_name[backslash_indices[-2]+1:backslash_indices[-1]]
                if feature_ext_name == 'wackerman' or feature_ext_name == 'AR':
                    break
                classifier_name = file_name[backslash_indices[-3]+1:backslash_indices[-2]]
                subj = file_name[underline_indices[-1]+1:-4]
    #            print feature_ext_name, classifier_name, subj
#                 print file_name
                npzfile = np.load(file_name)
                error = npzfile['error']
                accuracy = 100 - error * 100
                if dataset_name == 'SM2' or dataset_name == 'BCICIV1':
                    
#                     error = npzfile['error'][1]
#                     if subj != 'CS':
#                         print 4
                    if subj != 'CS' and subj != 'ID':
#                         print subj
                        mysum = 0
                        begin_ind = int(float(file_name[underline_indices[-3]+1:underline_indices[-2]])) - 1
                        if (len(npzfile['Y_pred']) + begin_ind) != len(subjects_true_labels_dict[subj]):
                            print 'error' 
                        
                        for element_id, element in enumerate(npzfile['Y_pred']):
                            
                            if element == subjects_true_labels_dict[subj][element_id+begin_ind]:
                                mysum += 1
#                         accuracy = 100.0 * mysum/float(len(npzfile['Y_pred']))
                        accuracy = precision_recall_fscore_support(subjects_true_labels_dict[subj][begin_ind:]-1, npzfile['Y_pred']-1, labels=[0,1])
                        accuracy = accuracy[0][1]
                    else:
                        accuracy = 0
                statistical_test_results[ classifiers_dict[classifier_name]*len(features_dict) + features_dict[feature_ext_name], subjects_dict[subj]] = accuracy
                statistical_test_results_titles[classifiers_dict[classifier_name]*len(features_dict) + features_dict[feature_ext_name]] = classifier_name + '.' + feature_ext_name
                results[classifiers_dict[classifier_name], features_dict[feature_ext_name],subjects_dict[subj]] = accuracy
                discarded_periods[classifiers_dict[classifier_name], features_dict[feature_ext_name],subjects_dict[subj]] = file_name[backslash_indices[-1]+1:underline_indices[2]]
    
    return results, sorted(classifiers_dict, key=classifiers_dict.get), sorted(features_dict, key=features_dict.get),  statistical_test_results, statistical_test_results_titles

if __name__ == '__main__':
    
    matplotlib.use('TkAgg')
    
    ##################################################################################################################
    dataset_name = 'BCICIII3b'
    accuracies, classifiers_list, features_list, statistical_test_accs1, stat_titles = read_optimal_accuracies(dataset_name, 'ALL')
#     accuracies.tofile('accuracies.csv')
#     loaded_accuracies = np.fromfile('accuracies.csv')
#     accuracies = np.reshape(loaded_accuracies, (7,5,3))
    subject_names = ['O3', 'S4', 'X11']
#     classifiers_list = ['Boosting', 'LogisticRegression', 'RANDOM_FOREST','SVM', 'LDA', 'QDA', 'MLP']
#     features_list = ['BP', 'AR', 'morlet', 'logbp', 'wackerman']
#     my_plotter.bar_plotter(classifiers_list, features_list, accuracies, subject_names)
#     my_plotter.bar_plotter(features_list, classifiers_list, accuracies, subject_names, type = 'COMPARE_FEATURES')
    
    
    first_accuracies = np.copy(accuracies)
    #####################################################################################################################
    dataset_name = 'BCICIV2b'
    accuracies, classifiers_list, features_list, statistical_test_accs2, stat_titles = read_optimal_accuracies(dataset_name, 'ALL')
#     accuracies.tofile('accuracies.csv')
#     loaded_accuracies = np.fromfile('accuracies.csv')
#     accuracies = np.reshape(loaded_accuracies, (7,5,3))
    subject_names = ['100', '200', '300', '400', '500', '600', '700', '800', '900']

#     my_plotter.bar_plotter(classifiers_list, features_list, accuracies, subject_names)
#     my_plotter.bar_plotter(features_list, classifiers_list, accuracies, subject_names, type = 'COMPARE_FEATURES')
    second_accuracies = np.copy(accuracies)
    
    combined_accuracies = np.append(first_accuracies, second_accuracies, axis = 2)
    
    ##########################################################################################################################
     #####################################################################################################################
    dataset_name = 'BCICIV2a'
    accuracies, classifiers_list, features_list, statistical_test_accs2, stat_titles = read_optimal_accuracies(dataset_name, 'CSP')
#     accuracies.tofile('accuracies.csv')
#     loaded_accuracies = np.fromfile('accuracies.csv')
#     accuracies = np.reshape(loaded_accuracies, (7,5,3))
    subject_names = ['1', '2', '3', '4', '5', '6', '7', '8', '9']

    my_plotter.bar_plotter(classifiers_list, features_list, accuracies, subject_names)
    my_plotter.bar_plotter(features_list, classifiers_list, accuracies, subject_names, type = 'COMPARE_FEATURES')
    third_accuracies = np.copy(accuracies)
    
    combined_accuracies = np.append(combined_accuracies, third_accuracies, axis = 2)
    ######################################################################################################################
#     bbb = combined_accuracies.reshape((12,-1))
#     for feature_ind, feature in enumerate(features_list):
#         np.savetxt( 'F:/results/' + feature + '_all_accuracies.csv', np.transpose(combined_accuracies[:, feature_ind, :]), delimiter = ',', header = ','.join(classifiers_list))
#     
#     statistical_test_accs = np.transpose(np.concatenate((statistical_test_accs1, statistical_test_accs2), axis = 1))
#     np.savetxt('F:/results/' + 'all_features_accuracies.csv', statistical_test_accs, delimiter = ',', header = ','.join(stat_titles))

    dataset_name = 'SM2'
    accuracies, classifiers_list, features_list, statistical_test_accs2, stat_titles = read_optimal_accuracies(dataset_name, 'CSP')

    subject_names = ["CB",'CS','ID','KT']

    my_plotter.bar_plotter(classifiers_list, features_list, accuracies, subject_names)
    
    ##########################################################################################################################
    dataset_name = 'BCICIV1'
    accuracies, classifiers_list, features_list, statistical_test_accs2, stat_titles = read_optimal_accuracies(dataset_name, 'CSP')
#     accuracies.tofile('accuracies.csv')
#     loaded_accuracies = np.fromfile('accuracies.csv')
#     accuracies = np.reshape(loaded_accuracies, (7,5,3))
    subject_names = ["a",'b','f','g']

    my_plotter.bar_plotter(classifiers_list, features_list, accuracies, subject_names)