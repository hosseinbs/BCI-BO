import numpy as np
import sys
import os
import json

sys.path.append('./BCI_Framework')
import Configuration_BCI
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


if __name__ == '__main__':
    
    ##################################################input values##########################################################################
    dataset = 'BCICIII3b'
    classifier_name = 'LogisticRegression'
    feature_extractor_name = 'BP'
    
    
    #########################################################################################################################################
    true_labels_folder = 'calc_results_labels/'
    config = Configuration_BCI.Configuration_BCI('BCI_Framework', dataset, 'ALL')
    
    opt_res_folder = os.path.join(config.configuration["results_opt_path_str"],classifier_name)
    opt_res_folder = os.path.join(opt_res_folder, feature_extractor_name)
    
    res_folder = os.path.join(config.configuration["results_path_str"],classifier_name)
    res_folder = os.path.join(res_folder, feature_extractor_name)
    
    opt_file_names = [ f for f in os.listdir(opt_res_folder) if os.path.isfile(os.path.join(opt_res_folder,f)) ]
    
#     O3_Y_train = np.array([1,2,2,1,1,1,2,1,1,2,1,2,1,1,2,1,2,1,2,2,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2,1,1,1,2,1,2,2,2,1,2,1,1,2,2,2,2,2,1,1,1,1,1,1,2,1,2,2,1,2,2,2,2,1,1,2,1,1,1,1,1,1,1,2,2,2,2,2,2,2,1,1,2,1,2,1,1,2,1,1,2,1,1,1,1,2,1,2,2,1,1,2,1,1,2,2,1,1,1,1,1,2,1,2,2,2,2,2,1,2,2,2,2,1,2,2,2,1,1,1,1,1,1,2,2,2,2,1,1,2,1,2,1,1,2,1,2,2,2,2,1,2,1,1,1,2,1,1,2,1,2,1,1,1,2,1,2,2,1,2,2,2,1,2,2,1,1,1,2,1,1,2,1,2,1,2,2,2,1,2,2,1,1,1,2,1,1,2,2,2,1,1,1,2,2,1,2,2,1,1,1,2,1,1,2,1,2,1,1,2,2,1,2,2,2,1,1,1,2,1,1,1,2,2,1,1,1])
#     S4_Y_train = np.array([2,2,1,2,2,2,1,2,1,2,1,2,1,1,1,1,2,1,1,2,2,1,2,1,1,2,2,1,2,2,1,2,2,1,2,1,2,2,1,1,2,2,1,2,1,1,1,1,2,1,2,1,2,2,1,1,1,1,2,2,1,2,1,2,1,2,2,2,2,1,2,2,1,1,2,1,2,1,1,1,2,1,2,2,1,1,2,1,2,2,2,1,2,2,2,1,2,1,1,1,1,2,1,2,1,2,2,1,2,2,1,1,2,1,1,2,1,2,2,2,2,2,2,2,2,1,2,2,1,1,2,1,2,1,1,2,1,1,1,1,1,1,2,2,2,2,1,2,2,2,1,2,1,2,2,1,1,1,2,2,1,2,2,1,1,1,1,1,2,1,2,1,2,2,2,2,1,2,2,1,2,1,2,2,1,1,1,1,1,1,1,2,1,2,1,1,1,2,2,1,2,2,2,1,1,2,1,1,1,2,2,2,2,2,2,1,2,1,1,1,2,2,2,2,2,2,1,1,1,1,2,1,1,2,1,2,1,2,2,2,2,1,2,2,2,1,1,1,2,2,1,2,2,2,1,1,2,2,2,2,1,1,2,1,1,2,1,1,2,2,1,2,2,2,1,2,2,1,2,2,1,2,1,1,2,1,1,1,2,2,2,1,1,2,1,1,1,1,2,2,1,2,1,2,1,2,1,1,2,2,2,2,2,2,1,1,2,2,2,1,1,2,2,1,1,1,2,2,2,1,1,1,2,1,2,2,2,1,1,2,1,2,2,1,1,2,1,1,1,1,2,1,1,2,2,2,2,1,1,2,1,2,1,1,2,1,1,1,1,1,2,1,1,1,1,2,1,2,2,2,1,1,2,2,1,1,1,2,2,1,1,2,2,2,2,1,2,1,2,1,1,1,2,2,1,1,2,1,1,1,1,2,1,2,1,1,1,2,1,2,2,2,1,1,1,1,1,1,2,1,1,1,1,2,2,2,2,1,1,2,1,1,2,2,1,1,2,1,2,1,1,2,2,1,1,2,1,1,2,2,2,2,2,1,1,2,1,2,2,2,1,2,1,2,2,2,1,1,2,2,2,1,2,1,2,2,2,1,2,2,1,1,1,1,1,2,1,1,1,2,1,1,1,2,2,2,1,1,2,2,1,1,1,2,1,1,2,2,2,2,2,2,1,1,2,1,1,2,2,2,1,1,1,2,1,1,2,2,2,1])
#     X11_Y_train = np.array([1,1,2,1,2,1,1,1,1,2,2,1,2,2,2,1,2,2,2,2,2,1,2,2,1,2,1,1,1,1,2,1,2,1,1,1,1,1,1,2,1,2,1,1,1,2,2,2,1,2,1,2,1,1,1,2,2,2,1,2,2,1,2,2,1,2,1,1,1,2,2,1,2,2,1,1,1,1,1,2,2,1,2,2,1,1,2,1,2,2,1,1,1,1,1,1,2,2,1,2,1,1,2,1,2,2,2,2,1,1,2,2,2,2,1,1,2,2,2,1,2,1,1,1,1,2,2,1,2,2,1,2,2,1,1,1,2,2,1,1,2,2,1,2,2,1,1,1,2,1,2,1,2,1,2,2,2,2,1,1,2,1,1,2,2,1,1,1,1,2,1,1,1,1,2,2,1,2,1,1,1,2,1,2,1,2,2,2,1,2,2,2,2,1,2,1,2,1,1,1,2,2,1,1,1,2,2,2,1,2,2,1,1,2,2,1,1,2,1,1,1,1,2,2,2,2,2,1,1,1,1,2,2,1,1,2,2,2,2,1,1,2,1,1,2,2,1,1,1,1,1,1,2,2,1,2,1,1,2,2,1,1,2,2,1,1,1,2,2,1,2,2,2,2,2,1,1,2,1,2,2,1,1,2,2,2,2,2,1,2,2,1,2,2,1,1,1,2,1,2,1,2,2,1,1,1,1,1,1,2,2,1,2,1,2,2,1,2,2,1,2,1,2,2,2,1,1,1,1,2,1,1,2,1,2,1,1,2,1,1,1,2,1,2,1,2,2,2,2,2,1,2,1,2,1,1,1,1,2,1,1,1,2,2,2,1,2,2,2,2,2,2,2,1,1,1,2,2,1,2,1,1,2,1,2,1,1,2,1,1,1,1,2,1,1,2,1,2,1,2,2,1,1,1,2,2,1,2,1,1,1,2,2,1,2,1,1,2,2,1,1,1,1,1,2,2,1,1,1,2,1,2,1,2,1,1,2,2,1,1,2,2,1,1,2,1,2,2,2,1,1,2,2,2,1,2,1,2,2,2,1,2,1,2,2,1,2,1,1,1,2,2,2,2,1,2,2,2,2,2,1,2,1,1,2,1,2,1,1,1,2,2,2,1,1,1,2,2,2,2,1,1,2,2,2,1,2,1,2,1,2,2,1,1,2,2,1,2,2,2,1,1,2,2,1,1,2,2,2,1,2,2,2,2,1,1,1,1,2,2])
    
    true_dict = {}
    all_subjects_test_probs_dict = dict()
    for subject in config.configuration["subject_names_str"]:
        
        true_labels = np.loadtxt(os.path.join(true_labels_folder, subject + '_Y_test.txt'))
        true_dict[subject] = true_labels 
        all_subjects_test_probs_dict[subject] = np.zeros((len(true_dict[subject]),2))
    
    for subject in config.configuration["subject_names_str"]:
        
        X_train = [] 
        X_test = []
        y_train = []
        cv_accs = []
        opt_accs = []
        for opt_file_name in opt_file_names:
            
            if (subject + '.npz') in opt_file_name:
                opt_file_path = os.path.join(opt_res_folder,opt_file_name)
                file_path = os.path.join(res_folder,opt_file_name)
                cv_file_path = file_path[0:-4]
                
                with open(cv_file_path,'r') as file:
                    all_res = json.load(file)
                    cv_acc = 1.0 - all_res['error']
                    
                npzfile = np.load(opt_file_path)
                probs_test = npzfile['probs_test']
                all_subjects_test_probs_dict[subject] += cv_acc * probs_test  
            
            
    for subject in config.configuration["subject_names_str"]:
            
        pred = np.argmax(all_subjects_test_probs_dict[subject], axis = 1)

        print 'subject ' + subject + ' weighted avg result: ', 100.0*sum(pred == true_dict[subject])/float(len(true_dict[subject]))
        
