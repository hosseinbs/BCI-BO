import numpy as np
import math
import os
import sys
import json

sys.path.append('./BCI_Framework')

import Main
import Single_Job_runner as SJR
import my_plotter
import os
import re
import Configuration_BCI
import matplotlib

def read_optimal_accuracies(dataset_name, CSPorALL):
    
    bcic = Main.Main('BCI_Framework', dataset_name,'RANDOM_FOREST', 'BP', CSPorALL, -1, 'python')
    
    opt_res_path = bcic.config.configuration['results_opt_path_str']
    cv_res_path = bcic.config.configuration['results_path_str']
    
    classifiers_dict = {'Boosting':0, 'LogisticRegression':1, 'RANDOM_FOREST':2,'SVM':3, 'LDA':4, 'QDA':5 , 'MLP':6}
    features_dict = {'BP':0, 'logbp':1, 'morlet':2}#, 'AR':3}
    
    results_opt = np.zeros((len(classifiers_dict),len(features_dict), bcic.config.configuration["number_of_subjects"]))
    statistical_test_results_opt = np.zeros((len(classifiers_dict) * len(features_dict), bcic.config.configuration["number_of_subjects"]))

    results_cv = np.zeros((len(classifiers_dict),len(features_dict), bcic.config.configuration["number_of_subjects"]))
    statistical_test_results_cv = np.zeros((len(classifiers_dict) * len(features_dict), bcic.config.configuration["number_of_subjects"]))

    statistical_test_results_titles = [None] * (len(classifiers_dict) * len(features_dict))
    
    discarded_periods = np.empty((len(classifiers_dict),len(features_dict), bcic.config.configuration["number_of_subjects"]), dtype='S10')
    subjects_dict = {}
    
    for ind, subj in enumerate(bcic.config.configuration["subject_names_str"]):
        subjects_dict.update({subj:ind})
        
    for dirname, dirnames, filenames in os.walk(opt_res_path):
        for filename in filenames:

            if filename[-4:] != '.pkl':
                file_name = os.path.join(dirname, filename)
                backslash_indices = [m.start() for m in re.finditer("\\\\", file_name)]
                underline_indices = [m.start() for m in re.finditer("_", file_name)]
                
                feature_ext_name = file_name[backslash_indices[-2]+1:backslash_indices[-1]]
                classifier_name = file_name[backslash_indices[-3]+1:backslash_indices[-2]]
                subj = file_name[underline_indices[-1]+1:-4]
                cv_file_path = os.path.join(cv_res_path, classifier_name + '/' + feature_ext_name + '/' + file_name[backslash_indices[-1]:-4])
    #            print feature_ext_name, classifier_name, subj
                # read optimal accuracy
                npzfile = np.load(file_name)
                
                if dataset_name == 'SM2' or dataset_name == 'BCICIV1':
                    error = float(npzfile['error'][2])
                    accuracy = 100 - error
                else:
                    error = npzfile['error']
                    accuracy = 100 - error * 100
                # read cv accuracy
                cv_res = json.load(open(cv_file_path))
                if feature_ext_name in features_dict.keys():
                    statistical_test_results_opt[ classifiers_dict[classifier_name]*len(features_dict) + features_dict[feature_ext_name], subjects_dict[subj]] = accuracy
                    statistical_test_results_titles[classifiers_dict[classifier_name]*len(features_dict) + features_dict[feature_ext_name]] = classifier_name + '.' + feature_ext_name
                    if classifier_name == 'RANDOM_FOREST':
                        statistical_test_results_titles[classifiers_dict[classifier_name]*len(features_dict) + features_dict[feature_ext_name]] = 'RANDOMFOREST.' + feature_ext_name
                    results_opt[classifiers_dict[classifier_name], features_dict[feature_ext_name],subjects_dict[subj]] = accuracy
                    discarded_periods[classifiers_dict[classifier_name], features_dict[feature_ext_name],subjects_dict[subj]] = file_name[backslash_indices[-1]+1:underline_indices[2]]
                    
                    statistical_test_results_cv[ classifiers_dict[classifier_name]*len(features_dict) + features_dict[feature_ext_name], subjects_dict[subj]] = 100 - 100.0*cv_res['error']
                    results_cv[classifiers_dict[classifier_name], features_dict[feature_ext_name],subjects_dict[subj]] = 100 - 100.0*cv_res['error']
    
    return results_opt, results_cv, sorted(classifiers_dict, key=classifiers_dict.get), sorted(features_dict, key=features_dict.get),  statistical_test_results_opt, statistical_test_results_cv, statistical_test_results_titles

if __name__ == '__main__':
    
    result_folder = "../../results_tables"
    matplotlib.use('TkAgg')
    dataset_names = ['BCICIII3b', 'BCICIV2b', 'BCICIV2a', 'SM2', 'BCICIV1']
    subject_names = [['O3', 'S4', 'X11'], ['100', '200', '300', '400', '500', '600', '700', '800', '900'],
                     ['1', '2', '3', '4', '5', '6', '7', '8', '9'], ['CB', 'CS', 'ID', 'KT'], ['a', 'b', 'f', 'g']]
    row_names = ['Dataset','O3', 'S4', 'X11', '100', '200', '300', '400', '500', '600', '700', '800', '900','1', '2', '3', '4', '5', '6', '7', '8', '9', 'CB', 'CS', 'ID', 'KT', 'a', 'b', 'f', 'g']
    ##################################################################################################################
    combined_accuracies = None
    feat_classifier_accuracies = None
    
    combined_accuracies_cv = None
    feat_classifier_accuracies_cv = None
    
    classifiers_list, features_list = None, None
    for data_ind, dataset in enumerate(dataset_names):
        accuracies_opt, accuracies_cv, classifiers_list, features_list, statistical_test_accs1_opt, statistical_test_accs1_cv, stat_titles1 = read_optimal_accuracies(dataset, 'ALL')
        if data_ind == 0:
            stat_titles = stat_titles1
            
        if combined_accuracies is None:
            combined_accuracies = np.copy(accuracies_opt)
            feat_classifier_accuracies = np.copy(statistical_test_accs1_opt)
            
            combined_accuracies_cv = np.copy(accuracies_cv)
            feat_classifier_accuracies_cv = np.copy(statistical_test_accs1_cv)
        else:
            combined_accuracies = np.append(combined_accuracies, np.copy(accuracies_opt), axis = 2)
            feat_classifier_accuracies = np.append(feat_classifier_accuracies, np.copy(statistical_test_accs1_opt), axis = 1)
            
            combined_accuracies_cv = np.append(combined_accuracies_cv, np.copy(accuracies_cv), axis = 2)
            feat_classifier_accuracies_cv = np.append(feat_classifier_accuracies_cv, np.copy(statistical_test_accs1_cv), axis = 1)
    
    
    for c_ind, classifier in enumerate(classifiers_list):
        if classifier == 'RANDOM_FOREST':
            classifiers_list[c_ind] = 'RANDOMFOREST' 
    
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    for feature_ind, feature in enumerate(features_list):
        np.savetxt( result_folder + '/' + feature + '_all_accuracies.csv', np.transpose(combined_accuracies[:, feature_ind, :]), fmt='%.2f', delimiter = ',', header = ','.join(classifiers_list))
        np.savetxt( result_folder + '/' + feature + '_all_accuracies_cv.csv', np.transpose(combined_accuracies_cv[:, feature_ind, :]), fmt='%.2f', delimiter = ',', header = ','.join(classifiers_list))
    
    for classifier_ind, classifier in enumerate(classifiers_list):
        np.savetxt( result_folder + '/'  + classifier + '_all_accuracies.csv', np.transpose(combined_accuracies[classifier_ind, :, :]), fmt='%.2f', delimiter = ',', header = ','.join(features_list))
        np.savetxt( result_folder + '/' + classifier + '_all_accuracies_cv.csv', np.transpose(combined_accuracies_cv[classifier_ind, :, :]), fmt='%.2f', delimiter = ',', header = ','.join(features_list))
    
#     statistical_test_accs = np.transpose(np.concatenate((statistical_test_accs1, statistical_test_accs2), axis = 1))
    np.savetxt(result_folder + '/'  + 'all_features_accuracies.csv', feat_classifier_accuracies.T, fmt='%.2f', delimiter = ',', header = ','.join(stat_titles))
    np.savetxt(result_folder + '/'  + 'all_features_accuracies_cv.csv', feat_classifier_accuracies_cv.T, fmt='%.2f', delimiter = ',', header = ','.join(stat_titles))
    
    
    for file in os.listdir(result_folder + '/' ):
        with open(os.path.join(result_folder,file),'r+') as f:
            all_results = f.readlines()
            all_results[0] = all_results[0][1:] 
            for res_ind, res in enumerate(all_results):
                all_results[res_ind] = row_names[res_ind] + ',' + res
            f.seek(0, 0)
            f.write("".join(all_results))
#             f.write(all_results)
