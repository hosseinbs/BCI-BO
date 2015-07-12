# from pylab import *

import numpy as np
import sys
import os
import json

sys.path.append('./BCI_Framework')
import Configuration_BCI
from Single_Job_runner import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn import cross_validation
from sklearn.gaussian_process import GaussianProcess

import BO_BCI
import spearmint_lite

def generate_param_list(Job_Params, params, config, subject):
    
    
    sp = spearmint_lite.spearmint_lite(Job_Params, [], config, Job_Params.type)
    params_dict = sp.generate_params_dict(params, subject)
    params_list = []
    
    if "cutoff_frequencies_low_list" in params_dict:
#             cutoff_frequencies_low_list = params_dict.pop('cutoff_frequencies_low_list')
#             cutoff_frequencies_high_list = params_dict.pop('cutoff_frequencies_high_list')
        params_list = [float(params_dict['discard_mv_begin']), float(params_dict['discard_mv_end']), float(params_dict['discard_nc_begin']), 
                       float(params_dict['discard_nc_end']), float(params_dict['window_size']), float(params_dict['window_overlap_size']), 
                        params_dict['cutoff_frequencies_low_list'], params_dict['cutoff_frequencies_high_list'], params_dict['channel_type']]
        
    else:
        params_list = [float(params_dict['discard_mv_begin']), float(params_dict['discard_mv_end']), float(params_dict['discard_nc_begin']),
                            float(params_dict['discard_nc_end']), float(params_dict['window_size']), float(params_dict['window_overlap_size']), params_dict['channel_type']]
            
    return params_list    

def write_results_to_file(input_matrix, write_type):
    if write_type == 'var':
        res_file = '../bo_results/res_var' + Job_Params.feature_extraction +'.csv'
    elif write_type == 'mean':
        res_file = '../bo_results/res_' + Job_Params.feature_extraction +'.csv'
    n_subjects = 9 #21
    final_results = np.zeros((n_subjects,4))
    final_results[:,0] = input_matrix[:,0]
    final_results[:,1] = np.max(input_matrix[:,1:5], axis = 1)
    final_results[:,2] = np.max(input_matrix[:,5:9], axis = 1)
    final_results[:,3] = np.max(input_matrix[:,9:], axis = 1)
    np.savetxt(res_file, final_results, delimiter = ',', fmt='%.2f', header = "Manual, GP, RF, Random")
    with open(res_file,'r+') as f:
        all_results = f.readlines()
        all_results[0] = all_results[0][1:] 
        all_results[0] = 'methods,' + all_results[0]
        for res_ind, res in enumerate(all_results[1:]):
            all_results[res_ind+1] = res_file_rows[res_ind] + ',' + res
        f.seek(0, 0)
        f.write("".join(all_results))

def write_results_to_file2(input_matrix_mean, input_matrix_var):
    
    max_indices = np.argmax(input_matrix_mean, axis = 1)
    
    res_file = '../bo_results/results_' + Job_Params.feature_extraction +'.csv'
    input_matrix = input_matrix_mean.astype(str) + "$\pm$" + input_matrix_var.astype(str) + "                       " 
    for row_ind, row in enumerate(input_matrix):
        row[max_indices[row_ind]] = "\\cellcolor{blue!25}" + row[max_indices[row_ind]] 
        input_matrix[row_ind] = row
    np.savetxt(res_file, input_matrix, fmt="%s",delimiter = ',', header = "Manual Search,MLR, VOTE, AVG, MIN, MLR, VOTE, AVG, MIN, MLR, VOTE, AVG, MIN")
    with open(res_file,'r+') as f:
        all_results = f.readlines()
        all_results[0] = all_results[0][1:] 
        all_results[0] = 'methods,' + all_results[0]
        for res_ind, res in enumerate(all_results[1:]):
            all_results[res_ind+1] = "subject" +  str(res_ind) + " (" + res_file_rows[res_ind] + '),' + res
        f.seek(0, 0)
        f.write("".join(all_results))


if __name__ == '__main__':
    
    ################################ first column is LR + BP and second column is LR + morlet    
    # framework_results = np.array([[80.5,82.39],[70.19,83.89],[74.26,78.15],[60.96,68.86],[56.33,58.37],[56.09,53.48],[94.79,94.79],[67.77,91.58],[75.7,82.87],
    #                  [53.02,72.84],[92.18,83.48],[77.55,86.12],[79,60.85],[61.13,54.42],[86.45,86.81],[73.68,41.23],[60.14,40.58],[56.74,26.98],
    #                  [87.36,56.32],[80.81,61.62],[83.71,39.39]])

    framework_results = np.array([[79,60.85],[61.13,54.42],[86.45,86.81],[73.68,41.23],[60.14,40.58],[56.74,26.98], [87.36,56.32],[80.81,61.62],[83.71,39.39]])

    ##################################################input values##########################################################################
    N_jobs = 55
    N_runs = 5
    chooser_modules = ["RandomForestEIChooser1", "GPEIOptChooser1", "RandomChooser1"]

        # ["RandomForestEIChooser1", "RandomForestEIChooser2", "RandomForestEIChooser3", "RandomForestEIChooser4", "RandomForestEIChooser5",
        #                "GPEIOptChooser1", "GPEIOptChooser2", "GPEIOptChooser3", "GPEIOptChooser4", "GPEIOptChooser5",
        #                "RandomChooser1","RandomChooser2", "RandomChooser3", "RandomChooser4", "RandomChooser5"]
    datasets = ['BCICIV2a']#['BCICIII3b', 'BCICIV2b', 'BCICIV2a']
    accuracy_types = ['MLR', 'MIN', 'VOTE', 'AVERAGE']
    n_subject_all_data = 9#21###############################################################################################
    all_datasets_final_results = np.zeros(shape = (n_subject_all_data, 12))
    all_datasets_final_results_dict = {"GPEIOptChooser":None, "RandomForestEIChooser":None, "RandomChooser":None}
    all_datasets_final_results_variances = np.zeros(shape = (n_subject_all_data, 12))
    n_subjects_processed = 0
    optimization_types_dict = {('BCICIII3b','BP'):[2], ('BCICIII3b','morlet'):[1], ('BCICIV2b','BP') : [2], ('BCICIV2b','morlet') : [1], ('BCICIV2a','BP') : [4]}#, ('BCICIV2a','morlet') : [3]}
    res_file_rows = []
    for data_ind, dataset in enumerate(datasets):
        config = Configuration_BCI.Configuration_BCI('BCI_Framework', dataset)
        all_datasets_final_results_dict = {"GPEIOptChooser":None, "RandomForestEIChooser":None, "RandomChooser":None}
        res_file_rows = res_file_rows + config.configuration['subject_names_str']
        for chooser_module in chooser_modules:     
            class Job_Params:
                    job_dir = '../Candidates'
                    num_all_jobs = N_jobs
                    dataset = dataset
                    seed = 1
                    classifier_name = 'LogisticRegression'
                    feature_extraction = 'BP'
                    n_concurrent_jobs = 1
                    chooser_module = chooser_module
                    type = optimization_types_dict[(dataset, feature_extraction)][0]
                    n_initial_candidates = 0
    
            chooser_module_for_dict = ''.join([i for i in Job_Params.chooser_module if not i.isdigit()])
            cv_folds = 5
            #########################################################################################################################################
            true_labels_folder = 'calc_results_labels/'
            
            opt_res_folder = os.path.join(config.configuration["results_opt_path_str"], Job_Params.classifier_name, Job_Params.feature_extraction, str(Job_Params.type), chooser_module_for_dict)
            
            res_folder = os.path.join(config.configuration["results_path_str"], Job_Params.classifier_name, Job_Params.feature_extraction, str(Job_Params.type), chooser_module_for_dict)
            
            opt_file_names = [ f for f in os.listdir(opt_res_folder) if os.path.isfile(os.path.join(opt_res_folder,f)) ]
        
            all_subjects_cv_errors = dict()
            true_dict = {}
            true_dict_train = {}
            all_subjects_accuracies_all_iterations = dict()
            all_subjects_test_probs_dict = dict()
            all_subjects_train_probs_dict = dict()
            all_subjects_accuracies_all_iterations_test = dict()
            all_subjects_accuracies_all_iterations_train = dict()
            
            for subject in config.configuration["subject_names_str"]:
                 
                true_labels = np.loadtxt(os.path.join(true_labels_folder, subject + '_Y_test.txt'))
                true_dict[subject] = true_labels
                
                true_labels_train = np.loadtxt(os.path.join(true_labels_folder, subject + '_Y_train.txt'))
                true_dict_train[subject] = true_labels_train 
                 
                for acc_type in accuracy_types:
                    all_subjects_accuracies_all_iterations_test[(subject,acc_type)] = []
                    all_subjects_accuracies_all_iterations_train[(subject,acc_type)] = []
                    
                ### MLR for test and train, print test accuracy for best train accuracy, MLR2 the same thing, Voting train and test,  print test accuracy for best train accuracy,
                ### the same thing for averaging, print all test accuracies after 40 iterations  
            for subject in config.configuration["subject_names_str"]:
                y = true_dict_train[subject]
                y_test = true_dict[subject] 
                skf = cross_validation.StratifiedKFold(y, n_folds=cv_folds)
                candidates_file_name = 'results_' + str(Job_Params.type) + '_' + Job_Params.chooser_module + '_'+ Job_Params.classifier_name + '_' + Job_Params.feature_extraction + '.dat_' + subject
                train_probs_till_now = [list() for _ in range(config.configuration['number_of_classes'])]
                test_probs_till_now = [list() for _ in range(config.configuration['number_of_classes'])]
                
                train_votes_till_now = []
                test_votes_till_now = []
                
                sum_test_probs = [list() for _ in range(config.configuration['number_of_classes'])]
                sum_train_probs = [list() for _ in range(config.configuration['number_of_classes'])]
                min_cv_error = 100
                min_cv_error_ind = 0
                with open(os.path.join('../Candidates', candidates_file_name),'r') as cand_file:
                    all_candidates = cand_file.readlines()
                for candidate_ind, candidate in enumerate(all_candidates):
                    if candidate_ind == N_jobs:
                        break
                    
                    if candidate.split()[0] == 'P':
                        break
                    
                    if min_cv_error > float(candidate.split()[0]):
                        min_cv_error = float(candidate.split()[0])
                        min_cv_error_ind = candidate_ind
                    
                    params_list = generate_param_list(Job_Params, candidate.split()[2:], config, subject)
                    out_name = Simple_Job_Runner.generate_learner_output_file_name(params_list, subject)
        #             print out_name
                    cv_file_name = os.path.join(res_folder, out_name)
                    opt_file_name = os.path.join(opt_res_folder, out_name + '.npz')
                      
                    if not os.path.exists(opt_file_name):
                        continue
                        
                    all_subjects_accuracies_all_iterations_train[(subject,'MLR')] = all_subjects_accuracies_all_iterations_train[(subject,'MLR')] + [0]
                    
                    npzfile = np.load(opt_file_name)
                    probs_test = npzfile['probs_test']
                    probs_train = npzfile['probs_train']
                    
                    test_votes_till_now.append(map(int,npzfile['Y_pred']))
                    train_votes_till_now.append(map(int, npzfile['Y_pred_train']))
                    
                    for class_ind in range(config.configuration['number_of_classes']):
                    
                        train_probs_till_now[class_ind].append(probs_train[:,class_ind])
                        test_probs_till_now[class_ind].append(probs_test[:,class_ind])
    
                    test_predictions = np.zeros(shape = (config.configuration['number_of_classes'], len(y_test)))
                    cv_res = [None] * cv_folds
    
        ###################################################################################Minimum Training Error####################################################################                
                    all_subjects_accuracies_all_iterations_train[(subject,'MIN')] = all_subjects_accuracies_all_iterations_train[(subject,'MIN')] + [min_cv_error]
                    all_subjects_accuracies_all_iterations_test[(subject, 'MIN')] = all_subjects_accuracies_all_iterations_test[(subject, 'MIN')] + [npzfile['error']] 
    
                    #########################################MLR########################################################################
                    
                    for class_ind in range(config.configuration['number_of_classes']):
                        
                        cv_fold = 0
                        X = np.array(train_probs_till_now[class_ind]).T
                        X_test = np.array(test_probs_till_now[class_ind]).T
                        
                        sum_test_probs[class_ind] = np.sum(X_test, axis = 1)
                        sum_train_probs[class_ind] = np.sum(X, axis = 1) 
     
                        for train_index, test_index in skf:
                            
                            X_train, X_test_cv = X[train_index], X[test_index]
                            y_train, y_test_cv = y[train_index], y[test_index]
                            
                            regr = linear_model.LinearRegression()
                            # Train the model using the training sets
                            regr.fit(X_train, y_train)
                            
                            cv_res[cv_fold] = np.mean((regr.predict(X_test_cv) - y_test_cv) ** 2)
                            cv_fold += 1
                            
    #                     all_subjects_cv_errors[subject][(candidate_ind+1) / 5 - 2] = np.mean(cv_res)
                        all_subjects_accuracies_all_iterations_train[(subject,"MLR")][-1] = all_subjects_accuracies_all_iterations_train[(subject,'MLR')][-1] + np.mean(cv_res)
                        
                        regr = linear_model.LinearRegression()
                        # Train the model using the training sets
                        regr.fit(X, y)
                        clf = linear_model.Lasso(alpha = 0, positive=True)
                        clf.fit(X,y)
                        
    #                     gp = GaussianProcess(corr='cubic', theta0=1e-2, thetaL=1e-4, thetaU=1e-1, random_start=100, nugget = 1e-10)
    #                     gp.fit(X, y)
    #                     test_predictions[class_ind,:] = regr.predict(X_test)
                        test_predictions[class_ind,:] = clf.predict(X_test)
    #                     test_predictions[class_ind,:] = gp.predict(X_test)
                    
                    accuracy_till_now = 100.0 * sum(np.argmax(test_predictions, axis= 0) == y_test)/float(len(y_test))
                    all_subjects_accuracies_all_iterations_test[(subject, 'MLR')] = all_subjects_accuracies_all_iterations_test[(subject, 'MLR')]  + [accuracy_till_now]
                    ######################################################VOTING########################################################################
    
                    if candidate_ind > 0:
                        vote_count = np.argmax(np.apply_along_axis(lambda X: np.bincount(X, minlength = 5), 0, np.array(test_votes_till_now)), axis = 0)
                                        
                        vote_count_train = np.argmax(np.apply_along_axis(lambda X: np.bincount(X, minlength = 5), 0, np.array(train_votes_till_now)), axis = 0) 
                    else:
                        vote_count = test_votes_till_now[0]
                        vote_count_train = train_votes_till_now[0]
                        
                    # accuracy_till_now = 100.0 * np.sum((np.array(vote_count) - 1) == y_test)/float(len(y_test))
                    # accuracy_till_now_train = 100.0 * np.sum((np.array(vote_count_train) - 1) == y)/float(len(y))
                    accuracy_till_now = 100.0 * np.sum((np.array(vote_count)) == y_test)/float(len(y_test))
                    accuracy_till_now_train = 100.0 * np.sum((np.array(vote_count_train)) == y)/float(len(y))
                    all_subjects_accuracies_all_iterations_train[(subject, 'VOTE')] = all_subjects_accuracies_all_iterations_train[(subject, 'VOTE')]  + [accuracy_till_now_train]
                    all_subjects_accuracies_all_iterations_test[(subject, 'VOTE')] = all_subjects_accuracies_all_iterations_test[(subject, 'VOTE')]  + [accuracy_till_now]
                    ######################################################AVERAGING########################################################################
                    accuracy_till_now = 100.0 * np.sum(np.argmax(np.array(sum_test_probs), axis = 0) == y_test-1)/float(len(y_test))
                    accuracy_till_now_train = 100.0 * np.sum(np.argmax(np.array(sum_train_probs), axis = 0) == y-1)/float(len(y))
                    
                    all_subjects_accuracies_all_iterations_train[(subject, 'AVERAGE')] = all_subjects_accuracies_all_iterations_train[(subject, 'AVERAGE')]  + [accuracy_till_now_train]
                    all_subjects_accuracies_all_iterations_test[(subject, 'AVERAGE')] = all_subjects_accuracies_all_iterations_test[(subject, 'AVERAGE')]  + [accuracy_till_now]
                        
        
#             res_file_rows = []
            final_results = np.zeros(shape = (4, len(config.configuration["subject_names_str"])))
        #     final_results = np.zeros(shape = (4, 2*len(config.configuration["subject_names_str"])))
            last_ind = -1 #Job_Params.num_all_jobs - 1
            for subj_ind, subject in enumerate(config.configuration["subject_names_str"]):
        
#                 res_file_rows = res_file_rows + [subject]
        #         res_file_rows = res_file_rows + [subject, subject + '_after_40_iterations']
                mlr1, mlr2 = all_subjects_accuracies_all_iterations_test[(subject, 'MLR')][np.argmin(all_subjects_accuracies_all_iterations_train[(subject, 'MLR')])], all_subjects_accuracies_all_iterations_test[(subject, 'MLR')][last_ind]
                vote1, vote2 = all_subjects_accuracies_all_iterations_test[(subject, 'VOTE')][np.argmax(all_subjects_accuracies_all_iterations_train[(subject, 'VOTE')])], all_subjects_accuracies_all_iterations_test[(subject, 'VOTE')][last_ind]
                avg1, avg2 = all_subjects_accuracies_all_iterations_test[(subject, 'AVERAGE')][np.argmax(all_subjects_accuracies_all_iterations_train[(subject, 'AVERAGE')])], all_subjects_accuracies_all_iterations_test[(subject, 'AVERAGE')][last_ind]
                min12 =  100 - 100*all_subjects_accuracies_all_iterations_test[(subject, 'MIN')][np.argmin(all_subjects_accuracies_all_iterations_train[(subject, 'MIN')])]
                
        #         final_results[:, subj_ind*2] = [mlr1, vote1, avg1, min12]
        #         final_results[:, subj_ind*2+1] = [mlr2, vote2, avg2, min12]
                final_results[:, subj_ind] = [mlr2, vote2, avg2, min12]
                
                print subject, np.argmin(all_subjects_accuracies_all_iterations_train[(subject, 'MLR')]), all_subjects_accuracies_all_iterations_test[(subject, 'MLR')][np.argmin(all_subjects_accuracies_all_iterations_train[(subject, 'MLR')])], all_subjects_accuracies_all_iterations_test[(subject, 'MLR')][last_ind]
                print subject, np.argmax(all_subjects_accuracies_all_iterations_train[(subject, 'VOTE')]), all_subjects_accuracies_all_iterations_test[(subject, 'VOTE')][np.argmax(all_subjects_accuracies_all_iterations_train[(subject, 'VOTE')])], all_subjects_accuracies_all_iterations_test[(subject, 'VOTE')][last_ind]
                print subject, np.argmax(all_subjects_accuracies_all_iterations_train[(subject, 'AVERAGE')]), all_subjects_accuracies_all_iterations_test[(subject, 'AVERAGE')][np.argmax(all_subjects_accuracies_all_iterations_train[(subject, 'AVERAGE')])], all_subjects_accuracies_all_iterations_test[(subject, 'AVERAGE')][last_ind]
                print subject, np.argmin(all_subjects_accuracies_all_iterations_train[(subject, 'MIN')]), 100 - 100*all_subjects_accuracies_all_iterations_test[(subject, 'MIN')][np.argmin(all_subjects_accuracies_all_iterations_train[(subject, 'MIN')])]
            
            
            
            if all_datasets_final_results_dict[chooser_module_for_dict] == None:
                all_datasets_final_results_dict[chooser_module_for_dict] = np.transpose(final_results)
            else:
                all_datasets_final_results_dict[chooser_module_for_dict] = np.dstack((all_datasets_final_results_dict[chooser_module_for_dict],np.transpose(final_results)))
                
                
                
            if "GPEIOptChooser" in Job_Params.chooser_module:
#                 all_datasets_final_results[n_subjects_processed:n_subjects_processed + config.configuration["number_of_subjects"], 0:4] = np.transpose(final_results)
                all_datasets_final_results[n_subjects_processed:n_subjects_processed + config.configuration["number_of_subjects"], 0:4] += np.transpose(final_results)
                if len(all_datasets_final_results_dict[chooser_module_for_dict].shape) > 2:
                    all_datasets_final_results_variances[n_subjects_processed:n_subjects_processed + config.configuration["number_of_subjects"], 0:4] = np.std(all_datasets_final_results_dict[chooser_module_for_dict], axis = 2)
            elif "RandomForestEIChooser" in Job_Params.chooser_module:
#                 all_datasets_final_results[n_subjects_processed:n_subjects_processed + config.configuration["number_of_subjects"], 4:8] += np.transpose(final_results)
#                 all_datasets_final_results_variances[n_subjects_processed:n_subjects_processed + config.configuration["number_of_subjects"], 4:8] = np.std(all_datasets_final_results_dict[chooser_module_for_dict], axis = 2)
                all_datasets_final_results[n_subjects_processed:n_subjects_processed + config.configuration["number_of_subjects"], 4:8] += np.transpose(final_results)
                if len(all_datasets_final_results_dict[chooser_module_for_dict].shape) > 2:
                    all_datasets_final_results_variances[n_subjects_processed:n_subjects_processed + config.configuration["number_of_subjects"], 4:8] = np.std(all_datasets_final_results_dict[chooser_module_for_dict], axis = 2)
            elif "RandomChooser" in Job_Params.chooser_module:
                all_datasets_final_results[n_subjects_processed:n_subjects_processed + config.configuration["number_of_subjects"], 8:12] += np.transpose(final_results)
                if len(all_datasets_final_results_dict[chooser_module_for_dict].shape) > 2:
                    all_datasets_final_results_variances[n_subjects_processed:n_subjects_processed + config.configuration["number_of_subjects"], 8:12] = np.std(all_datasets_final_results_dict[chooser_module_for_dict], axis = 2)
        
        ##### TODO: dont forget to divide by the number of seeds
        all_datasets_final_results[n_subjects_processed:n_subjects_processed + config.configuration["number_of_subjects"], :] = all_datasets_final_results[n_subjects_processed:n_subjects_processed + config.configuration["number_of_subjects"],:] #/ 5.0
#         all_datasets_final_results[n_subjects_processed:n_subjects_processed + config.configuration["number_of_subjects"], 4:8] = all_datasets_final_results[n_subjects_processed:n_subjects_processed + config.configuration["number_of_subjects"], 4:8] / 5.0
        n_subjects_processed += config.configuration["number_of_subjects"]
  
  
#     write_results_to_file(all_datasets_final_results, 'mean')
#     write_results_to_file(all_datasets_final_results_variances, 'var')
    
    if Job_Params.feature_extraction == "BP":
        all_datasets_final_results = np.column_stack((framework_results[:,0], all_datasets_final_results))
        all_datasets_final_results_variances = np.column_stack((np.zeros((n_subject_all_data, 1)), all_datasets_final_results_variances))
    elif Job_Params.feature_extraction == "morlet":
        all_datasets_final_results = np.column_stack((framework_results[:,1], all_datasets_final_results))
        all_datasets_final_results_variances = np.column_stack((np.zeros((n_subject_all_data, 1)), all_datasets_final_results_variances))
    # write_results_to_file2(all_datasets_final_results, all_datasets_final_results_variances)
    write_results_to_file(all_datasets_final_results, 'mean')
    
    
