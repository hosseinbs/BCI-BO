from pylab import *

import numpy as np
import sys
import os
import json

sys.path.append('./BCI_Framework')
import Configuration_BCI
from Single_Job_runner import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def generate_param_list(params, type):
    
    params_list = []
    if type == 1:
        params_list = [float(params[0]), float(params[1]), 0.0, 0.0, -1.0, 0.0]
    
    return params_list    


if __name__ == '__main__':
    
    ##################################################input values##########################################################################
#     dataset = 'BCICIII3b'
    dataset = 'BCICIV2b'
    classifier_name = 'LogisticRegression'
    feature_extractor_name = 'morlet'
    type = 1
    
    #########################################################################################################################################
    true_labels_folder = 'calc_results_labels/'
    config = Configuration_BCI.Configuration_BCI('BCI_Framework', dataset, 'ALL')
    
    opt_res_folder = os.path.join(config.configuration["results_opt_path_str"],classifier_name)
    opt_res_folder = os.path.join(opt_res_folder, feature_extractor_name)
    
    res_folder = os.path.join(config.configuration["results_path_str"],classifier_name)
    res_folder = os.path.join(res_folder, feature_extractor_name)
    
    opt_file_names = [ f for f in os.listdir(opt_res_folder) if os.path.isfile(os.path.join(opt_res_folder,f)) ]

    true_dict = {}
    all_subjects_accuracies_all_iterations = dict()
    all_subjects_test_probs_dict = dict()
    for subject in config.configuration["subject_names_str"]:
         
        true_labels = np.loadtxt(os.path.join(true_labels_folder, subject + '_Y_test.txt'))
        true_dict[subject] = true_labels 
        all_subjects_test_probs_dict[subject] = np.zeros((len(true_dict[subject]),2))
        all_subjects_accuracies_all_iterations[subject] = []
        
    for subject in config.configuration["subject_names_str"]:
        candidates_file_name = 'results_' + dataset + ".dat_" + subject
        with open(os.path.join('BCI_Framework', candidates_file_name),'r') as cand_file:
            all_candidates = cand_file.readlines()
        for candidate in all_candidates:
            params_list = generate_param_list(candidate.split()[2:],1)
            out_name = Simple_Job_Runner.generate_learner_output_file_name(params_list, subject)
#             print out_name
            cv_file_name = os.path.join(res_folder, out_name)
            opt_file_name = os.path.join(opt_res_folder, out_name + '.npz')
            if candidate[0] != 'P':
                with open(cv_file_name,'r') as cv_file:
                    all_res = json.load(cv_file)
                    cv_acc = 1.0 - all_res['error']
    #             print cv_acc
            if os.path.exists(opt_file_name):
                npzfile = np.load(opt_file_name)
                probs_test = npzfile['probs_test']
                all_subjects_test_probs_dict[subject] += cv_acc * probs_test
    
                pred = np.argmax(all_subjects_test_probs_dict[subject], axis = 1)
                accuracy_till_now = 100.0*sum(pred == true_dict[subject])/float(len(true_dict[subject]))
    #             print 'subject ' + subject + ' weighted avg result: ', accuracy_till_now
                all_subjects_accuracies_all_iterations[subject].append(accuracy_till_now)
        
        print all_subjects_accuracies_all_iterations[subject]

        plot(range(len(all_subjects_accuracies_all_iterations[subject])), all_subjects_accuracies_all_iterations[subject])

show()
