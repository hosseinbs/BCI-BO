import os
import sys
os.chdir('../bci_framework')
sys.path.append('./BCI_Framework')

import Main
import Configuration_BCI

import Single_Job_runner as SJR
import numpy as np
from os import listdir
from os.path import isfile, join
import json


def write_candidate_to_file(new_candidate, BO_type, classifier, feature, subject):
    
    for chooser_module in chooser_modules:
        bo_candidates_file_name = "results_" + str(BO_type) +"_" + chooser_module + "_" + classifier + "_" + feature + ".dat_" + subject
        bo_file_path = os.path.join('BCI_Framework' , bo_candidates_file_name)
        cv_results = json.load(open(os.path.join(res_path, res_file)))
    
        new_candidate = np.transpose(np.array([cv_results['error']] + new_candidate_list))
        if os.path.isfile(bo_file_path):
            candidates = np.loadtxt(bo_file_path)
            candidates = np.vstack((candidates,new_candidate))
            np.savetxt(bo_file_path, candidates, fmt='%.2f', delimiter=' ')

        else:
            print new_candidate
            np.savetxt(bo_file_path, new_candidate[None], fmt='%.2f', delimiter=' ')
            
            

if __name__ == '__main__':
    
    print "read initial candidates from framework results"
    
    BO_types = [1, 2, 3, 4]
    features = ['BP', 'morlet']
    classifier = 'LogisticRegression'
    datasets = [ 'BCICIII3b', 'BCICIV2b', 'BCICIV2a']
    
    chooser_modules = ["RandomForestEIChooser1", "RandomForestEIChooser2", "RandomForestEIChooser3", "RandomForestEIChooser4"]
    
    for dataset in datasets:
          
        config = Configuration_BCI.Configuration_BCI('BCI_Framework', dataset)
        base_res_path = config.configuration['results_path_str']
  
        for feature in features:
            res_path = os.path.join(base_res_path, classifier, feature)
            res_files = [ f for f in listdir(res_path) if isfile(os.path.join(res_path,f)) ]
              
            for res_file in res_files:
                params = res_file.split('_')
                  
                subject = params[-1]
                  
                params_list = [float(params[0]), float(params[1]), float(params[2]),
                       float(params[3]), float(params[4]), float(params[5]), params[6]]
#                 params_list = [float(params_dict['discard_mv_begin']), float(params_dict['discard_mv_end']), float(params_dict['discard_nc_begin']),
#                        float(params_dict['discard_nc_end']), float(params_dict['window_size']), float(params_dict['window_overlap_size']), params_dict['channel_type']]
                print params_list
                channel_type = params[6].split('-')[0]
                  
                channels_list_names = ['ALL', 'CSP2', 'CSP4', 'CSP6', 'CS']
                channels_list_numbers = [0, 1, 2, 3, 4]
                  
                channel_type = channels_list_names.index(channel_type)
                  
#####################################################################BOTYPE = 1################################################################################                
                if (dataset == 'BCICIV2b' or dataset == 'BCICIII3b') and feature == 'morlet':
                    BO_type = 1
                    new_candidate_list = [ 100, int(params_list[0]), int(params_list[1])]
                    write_candidate_to_file(new_candidate_list, BO_type, classifier, feature, subject)
                      
#####################################################################BOTYPE = 1################################################################################                
                elif (dataset == 'BCICIV2b' or dataset == 'BCICIII3b') and feature != 'morlet':
                    BO_type = 2
                    new_candidate_list = [ 100, int(params_list[0]), int(params_list[1]), 8, 12, 16, 24]
                    write_candidate_to_file(new_candidate_list, BO_type, classifier, feature, subject)
  
#####################################################################BOTYPE = 1################################################################################                
                if dataset == 'BCICIV2a' and feature == 'morlet':
                    BO_type = 3
                    new_candidate_list = [ 100, int(params_list[0]), int(params_list[1]), channel_type]
                    write_candidate_to_file(new_candidate_list, BO_type, classifier, feature, subject)
#####################################################################BOTYPE = 1################################################################################
                elif dataset == 'BCICIV2a' and feature != 'morlet':                
                    BO_type = 4
                    new_candidate_list = [ 100, int(params_list[0]), int(params_list[1]),  8, 12, 16, 24, channel_type]
                    write_candidate_to_file(new_candidate_list, BO_type, classifier, feature, subject)


######################################################################MOVE RESULTS TO THEIR FOLDERS##########################################################
    import shutil
    classifier = 'LogisticRegression'
    optimization_types_dict = {('BCICIII3b','BP'):[2], ('BCICIII3b','morlet'):[1], ('BCICIV2b','BP') : [2], ('BCICIV2b','morlet') : [1], ('BCICIV2a','BP') : [4], ('BCICIV2a','morlet') : [3]}
#     BO_selection_types = ["GPEIOptChooser", "RandomChooser1", "RandomForestEIChooser"]
    BO_selection_types = ["RandomChooser1", "RandomChooser2", "RandomChooser3", "RandomChooser4", "RandomChooser5"]
    
    for dataset in datasets:
        
        config = Configuration_BCI.Configuration_BCI('BCI_Framework', dataset)
        base_res_path = config.configuration['results_path_str']

        for feature in features:
            source_res_path = os.path.join(base_res_path, classifier, feature)
            for opt_type in optimization_types_dict[(dataset,feature)]:
                for bo_type in BO_selection_types:
                    res_files = [ f for f in listdir(source_res_path) if isfile(os.path.join(source_res_path,f)) ]
                    destination_res_path = os.path.join(source_res_path, str(opt_type), bo_type)
                    if not os.path.exists(destination_res_path):
                        os.makedirs(destination_res_path)
                    print 'source: ' + source_res_path
                    print 'destination: ' + destination_res_path
                    
                    for f in res_files:
                        splited_file_name = f.split('_')
                        if opt_type == 2 or opt_type == 4:
                            dest_file_name = '_'.join(splited_file_name[0:-2] + ['8.0','16.0', '12.0', '24.0'] + splited_file_name[-2:])
                            shutil.copy(os.path.join(source_res_path, f), os.path.join(destination_res_path,dest_file_name))
                        else:
                            shutil.copy(os.path.join(source_res_path, f), os.path.join(destination_res_path,f))
                    

