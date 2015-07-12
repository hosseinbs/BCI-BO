import os
import sys
sys.path.append('./BCI_Framework')

import Main
import Single_Job_runner as SJR
import numpy as np
import Configuration_BCI
import my_plotter

if __name__ == '__main__':

    classifiers_list= ['Boosting', 'LogisticRegression_l1', 'LogisticRegression_l2', 'RANDOM_FOREST','SVM_linear', 'SVM_rbf']
    classifiers_list_plot= ['Boosting', 'Logistic_R(l1)', 'Logistic_R(l2)', 'Random_Forest','SVM(l)', 'SVM(rbf)']
    features_list = ['BP', 'logbp', 'wackerman']

    def create_accuracies_matrix(classifiers_list, features_list, dataset):
        
        
        classifiers_dict = create_dict(classifiers_list)
        features_dict = create_dict(features_list)
        config = Configuration_BCI.Configuration_BCI('BCI_Framework', dataset)

        subjects_dict = create_dict(config.configuration["subject_names_str"])

        opt_error_matrix = np.zeros(shape = (len(classifiers_dict),len(features_dict), len(subjects_dict)))
        cv_error_matrix = np.zeros((len(classifiers_dict),len(features_dict), len(subjects_dict)))
        n_csps = -1
        if config.configuration['number_of_channels'] > 5:
            n_csps = 1

        
        for classifier in classifiers_dict.keys():
            for feature in features_dict.keys():
                bcic = Main.Main('BCI_Framework', dataset, classifier, feature, 'ALL', n_csps, 'python')
                
                for subject in bcic.config.configuration['subject_names_str']:
                    temp, temp, cv_err = bcic.find_learners_optimal_params(subject)
                    opt_error = bcic.find_opt_error(subject)
                    opt_error_matrix[classifiers_dict[classifier], features_dict[feature], subjects_dict[subject]] = opt_error
                    cv_error_matrix[classifiers_dict[classifier], features_dict[feature], subjects_dict[subject]] = cv_err
                    
                    
        return cv_error_matrix, opt_error_matrix
    
    
    def create_dict(my_list):
        my_dict = {}
        for ind, subj in enumerate(my_list):
            my_dict.update({subj:ind})   
        return my_dict      
    
    
    
    
    
    datasets = ['BCICIV2a','BCICIV2a', 'BCICIII3b']
    bar_chart_mat_opt = np.zeros((len(classifiers_list),len(features_list)))
    bar_chart_mat_cv = np.zeros((len(classifiers_list),len(features_list)))
                                 
    for dataset in datasets:
        
        cv_error_matrix, opt_error_matrix = create_accuracies_matrix(classifiers_list, features_list, dataset)
        classifiers_dict = create_dict(classifiers_list)
        features_dict = create_dict(features_list)
        config = Configuration_BCI.Configuration_BCI('BCI_Framework', dataset)

        subjects_dict = create_dict(config.configuration["subject_names_str"])
        
        for subject in config.configuration['subject_names_str']:
##            print subject 
##            print opt_error_matrix[:,:,subjects_dict[subject]]
#            
            min_indices = np.argmin(opt_error_matrix[:,:,subjects_dict[subject]], axis = 0)
            mins = np.min(opt_error_matrix[:,:,subjects_dict[subject]], axis = 0)
            mins_boolean = opt_error_matrix[:,:,subjects_dict[subject]] == mins
            
            
            for ind, min_ind in enumerate(min_indices):
                bar_chart_mat_opt[mins_boolean[:,ind],ind] +=1
#                bar_chart_mat_opt[min_ind,ind] += 1
                
#        for subject in config.configuration['subject_names_str']:
#            print subject 
#            print cv_error_matrix[:,:,subjects_dict[subject]]
            
#            min_indices = np.argmin(cv_error_matrix[:,:,subjects_dict[subject]], axis = 0)
#            for ind, min_ind in enumerate(min_indices):
#                bar_chart_mat_cv[min_ind,ind] += 1
                
        
        
    bar_chart_mat_opt = 100 * bar_chart_mat_opt / np.sum(bar_chart_mat_opt, axis = 0)
    print bar_chart_mat_opt
    
    my_plotter.bar_chart_plotter(bar_chart_mat_opt, classifiers_list_plot, features_list)
    
#    print bar_chart_mat_cv
    
    
#    print cv_error_matrix