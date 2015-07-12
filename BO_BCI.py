import spearmint_lite
import os
import sys
os.chdir('../bci_framework')
sys.path.append('./BCI_Framework')

import Main
import Configuration_BCI

import Single_Job_runner as SJR
import numpy as np
import itertools

from time import sleep
import sklearn

def generate_window_list_freq(window_start_list, window_length_list, limit):
    
    window_list = []
    for window_start_point in window_start_list:
        for window_length in window_length_list:
            if window_start_point + window_length <= limit:
                window_list.append([window_start_point, window_start_point + window_length])
    return window_list

def generate_window_list_time(window_start_list, window_length_list, limit):
    
    window_list = []
    for window_start_point in window_start_list:
        for window_length in window_length_list:
            if window_start_point + window_length <= limit:
                window_list.append([window_start_point, limit - (window_start_point + window_length)])
    return window_list

def generate_channel_indices(n_channels):
    
    indices_list = []
    for i in range(1,5):#9 is ok for ubc-pc
        indices_list = indices_list + list(itertools.combinations(range(n_channels),i))

    return indices_list

def revise_candidates(raw_candidates, BO_type):
    
    #if elemetn >1 and type 5 binary
    candidates = []
    
    for raw_candidate in raw_candidates:
        candidate = []
        for element_ind, element in enumerate(raw_candidate):
            if element_ind >= 1 and BO_type == 5:
                cand = np.zeros(22)
                cand[list(element)] = 1
                
                candidate = candidate + [int(''.join(map(str,map(int,cand))),2)]
            else:
                for el in element:
                    candidate.append(el)
            
        candidates.append(candidate)
    
    return candidates

def read_initial_candidates(file_name, config):
    """ """
    with open(file_name, 'r') as init_file:
        all_initial_cands = init_file.readlines() 
    for line_ind, line in enumerate(all_initial_cands):
        all_initial_cands[line_ind] = line.strip().split()
    
        all_initial_cands[line_ind] = map(lambda X: float(X) if X.replace('.','').replace('-','').isdigit() else X, all_initial_cands[line_ind])
    
#     all_initial_cands = np.loadtxt(file_name)
#     all_initial_cands = np.ndarray.tolist(all_initial_cands)
    ###################################3
    for init_cand_index in range(len(all_initial_cands)):
        all_initial_cands[init_cand_index][0] *= config.configuration['sampling_rate'] 
        all_initial_cands[init_cand_index][1] *= config.configuration['sampling_rate']
        if all_initial_cands[init_cand_index][1] < 0:
            all_initial_cands[init_cand_index][1] = -1 * all_initial_cands[init_cand_index][1]
                 
    ######################################
    return all_initial_cands

def read_initial_candidates_type5(file_name, config):
    """ """
    all_initial_cands = np.loadtxt(file_name)
    new_initial_cands = np.zeros(shape = (len(all_initial_cands), 3))
    all_initial_cands = np.ndarray.tolist(all_initial_cands)
    new_initial_cands = np.ndarray.tolist(new_initial_cands)
    ###################################3
    for init_cand_index in range(len(all_initial_cands)):
        new_initial_cands[init_cand_index][0] = all_initial_cands[init_cand_index][0] * config.configuration['sampling_rate'] 
        new_initial_cands[init_cand_index][1] = all_initial_cands[init_cand_index][1] * config.configuration['sampling_rate']
        if all_initial_cands[init_cand_index][1] < 0:
            new_initial_cands[init_cand_index][1] = -1 * all_initial_cands[init_cand_index][1]
        new_initial_cands[init_cand_index][2] = int(''.join(map(str, map(int,all_initial_cands[init_cand_index][2:]))), 2)
    ######################################
    return new_initial_cands


def generate_all_candidates(dataset_name, optimization_type):
    
    ##########################################################################################################################################################
    # the following code block generates potential candidates for Bayesian Optimizer
    
    # different BO types
    # type 1 -> only search for time window
    # type 2 -> search for time and frequency window
    # type 3 -> search for time window  and channels
    # type 4 -> search for time window and search for frequency window and channels
    # type 3-1 -> search for time window and search for frequency window for each channel separately
    # type 5 -> search for time window and search for frequency window for each channel separately     
    BO_type = optimization_type
    config = Configuration_BCI.Configuration_BCI('BCI_Framework', dataset_name)
    
    sampling_rate = config.configuration['sampling_rate']
    all_subjects_candidates_list = []
    
    for subj_ind, subj in  enumerate(config.configuration['subject_names_str']):
        
        mv_size = config.configuration['movement_trial_size_list'][subj_ind]
        all_mvs_have_same_lebngth = all(map(lambda x: x == mv_size, config.configuration['movement_trial_size_list']))
        n_channels = config.configuration['number_of_channels']
        
        candidates_list = []
        
        if BO_type == 1:
            
            window_start_list = map(int, np.arange(0, mv_size - sampling_rate, sampling_rate/4.0))
            window_length_list = map(int, np.arange(0.75 * sampling_rate, mv_size+1, sampling_rate/4.0)) 
            window_list = generate_window_list_time(window_start_list, window_length_list, mv_size)
            candidates_list = window_list 
    
#             all_initial_cands = read_initial_candidates('BCI_Framework/BO_type1_initial_candidates.txt', config)
        
        elif BO_type == 2:
            
            window_start_list = map(int, np.arange(0, mv_size - sampling_rate, sampling_rate/4.0))
            window_length_list = map(int, np.arange(0.75 * sampling_rate, mv_size+1, sampling_rate/4.0)) 
            window_list_mv = generate_window_list_time(window_start_list, window_length_list, mv_size)
            candidates_list.append(window_list_mv)
            
#             window_start_list_freq = np.arange(2, 6, 0.5)
#             window_length_list_freq = np.arange(26, 30, 0.75)
#             window_list_freq = generate_window_list_freq(window_start_list_freq, window_length_list_freq, 32)
#             window_list_freq = zip(window_list_freq, window_list_freq)
#             window_list_freq = revise_candidates(window_list_freq, 0)
            window_list_freq = []
            alpha_beta_candidates = []
            
            window_start_list_freq = np.arange(7, 9.5, 0.5)
            window_length_list_freq = np.arange(4, 6, 0.75)
            window_list_freq_alpha = generate_window_list_freq(window_start_list_freq, window_length_list_freq, 14)
            alpha_beta_candidates.append(window_list_freq_alpha)
            
            window_start_list_freq = np.arange(15, 19, 0.5)
            window_length_list_freq = np.arange(5, 10, 0.75)
            window_list_freq_beta = generate_window_list_freq(window_start_list_freq, window_length_list_freq, 26)
            alpha_beta_candidates.append(window_list_freq_beta)
             
            raw_candidates_alpha_beta = list(itertools.product(*alpha_beta_candidates))
            candidates_list_alpha_beta = revise_candidates(raw_candidates_alpha_beta, BO_type)

            window_list_freq = window_list_freq + candidates_list_alpha_beta
            
            candidates_list.append(window_list_freq)
            
#             candidates_list.append(candidates_list_alpha_beta)
#             all_initial_cands = read_initial_candidates('BCI_Framework/BO_type2_initial_candidates.txt', config)
                
        elif BO_type == 3:
            
            window_start_list = map(int, np.arange(0, mv_size - sampling_rate, sampling_rate/4.0))
            window_length_list = map(int, np.arange(0.75 * sampling_rate, mv_size+1, sampling_rate/4.0)) 
            window_list_mv = generate_window_list_time(window_start_list, window_length_list, mv_size)
            candidates_list.append(window_list_mv)
            
#             channels_list = [['ALL'], ['CSP2'], ['CSP4'], ['CSP6'], ['CS']]
            channels_list = [[0], [1], [2], [3], [4]]#???????????????????????????????????????????????????????????????????????????????????????????
            candidates_list.append(channels_list)
#             all_initial_cands = read_initial_candidates('BCI_Framework/BO_type3_initial_candidates.txt', config)
            
        elif BO_type == 4:
            
            window_start_list = map(int, np.arange(100, mv_size - sampling_rate, sampling_rate/4.0))##
            window_length_list = map(int, np.arange(0.75 * sampling_rate, mv_size+1, sampling_rate/4.0)) 
            window_list_mv = generate_window_list_time(window_start_list, window_length_list, mv_size)
            candidates_list.append(window_list_mv)
            
            window_start_list_freq = np.arange(2, 6, 0.5)
            window_length_list_freq = np.arange(26, 30, 0.75)
            window_list_freq = generate_window_list_freq(window_start_list_freq, window_length_list_freq, 32)
            window_list_freq = zip(window_list_freq, window_list_freq)
            window_list_freq = revise_candidates(window_list_freq, 0)
            
            alpha_beta_candidates = []
            
            window_start_list_freq = np.arange(7, 9.5, 0.5)
            window_length_list_freq = np.arange(4, 6, 0.75) 
            window_list_freq_alpha = generate_window_list_freq(window_start_list_freq, window_length_list_freq, 14)
            alpha_beta_candidates.append(window_list_freq_alpha)
            
            window_start_list_freq = np.arange(15, 19, 0.5)
            window_length_list_freq = np.arange(5, 10, 0.75) 
            window_list_freq_beta = generate_window_list_freq(window_start_list_freq, window_length_list_freq, 26)
            alpha_beta_candidates.append(window_list_freq_beta)
             
            raw_candidates_alpha_beta = list(itertools.product(*alpha_beta_candidates))
            candidates_list_alpha_beta = revise_candidates(raw_candidates_alpha_beta, BO_type)

            window_list_freq = window_list_freq + candidates_list_alpha_beta
            
            candidates_list.append(window_list_freq)
            
            channels_list = [[0], [1], [2], [3], [4]]#???????????????????????????????????????????????????????????????????????????????????????????
            candidates_list.append(channels_list)
            
#             all_initial_cands = read_initial_candidates('BCI_Framework/BO_type4_initial_candidates.txt', config)
            
        
#         elif BO_type == 5:
#             
#             window_start_list = map(int, np.arange(0, mv_size - sampling_rate, sampling_rate/4.0))
#             window_length_list = map(int, np.arange(0.75 * sampling_rate, mv_size+1, sampling_rate/4.0)) 
#             window_list_mv = generate_window_list_time(window_start_list, window_length_list, mv_size)
#             candidates_list.append(window_list_mv)
#     
#             indices_list = generate_channel_indices(n_channels)
#             candidates_list.append(indices_list)
#             all_initial_cands = read_initial_candidates('BCI_Framework/BO_type4_initial_candidates.txt', config)

        if BO_type != 1:
            raw_candidates = list(itertools.product(*candidates_list))
            candidates_list = revise_candidates(raw_candidates, BO_type)
        
#         candidates_list = all_initial_cands + candidates_list ################################################
#         Job_Params.n_initial_candidates = len(all_initial_cands) ############################################
        
        all_subjects_candidates_list.append(candidates_list)     
        if all_mvs_have_same_lebngth:
            break
        
    ##########################################################################################################################################################
    return all_subjects_candidates_list, 0
    

if __name__ == '__main__':

    print "Bayesian Optimization for BCI"
    
    datasets = ['BCICIV2a']#['BCICIII3b', 'BCICIV2b',
    classifier = 'LogisticRegression'
    feature = 'BP' #'morlet'] #morlet for type 2 and 4 does not work!!!!!
    optimization_types_dict = {('BCICIII3b','BP'):[2], ('BCICIII3b','morlet'):[1], ('BCICIV2b','BP') : [2], ('BCICIV2b','morlet') : [1], ('BCICIV2a','BP') : [4], ('BCICIV2a','morlet') : [3]}
    BO_selection_types = [ "GPEIOptChooser1", "RandomForestEIChooser1", "RandomChooser1"]

        # , "GPEIOptChooser2", "GPEIOptChooser3", "GPEIOptChooser4", "GPEIOptChooser5", "RandomForestEIChooser1", "RandomForestEIChooser2", "RandomForestEIChooser3", "RandomForestEIChooser4",
        #                   "RandomForestEIChooser5", "RandomChooser1", "RandomChooser2", "RandomChooser3", "RandomChooser4", "RandomChooser5"]
    
    all_subjects_candidates_dict = {}
    for dataset_ind, dataset in enumerate(datasets):
#         for bo_type in BO_selection_types:
        for optimization_type in optimization_types_dict[(dataset, feature)]:
            print dataset, feature, optimization_type
            all_subjects_candidates_dict[(dataset_ind, optimization_type)], _ = generate_all_candidates(dataset, optimization_type)    
    
    first_iteration = True
    finished = {}

    while True:
        
        if first_iteration == False and all(finished.values()): 
            break
        
        for dataset_ind, dataset in enumerate(datasets):
            for bo_type_ind, bo_type in enumerate(BO_selection_types):
                for optimization_type in optimization_types_dict[(dataset, feature)]:
                    print dataset, feature, bo_type, optimization_type

            
                    class Job_Params:
                        job_dir = '../Candidates'
                        num_all_jobs = 100
                        dataset = dataset
                        
                        from random import randrange
                        seed = randrange(50)
                        classifier_name = classifier
                        feature_extraction = feature
                        n_concurrent_jobs = 1
                        chooser_module = bo_type
                
                    Job_Params.n_initial_candidates = 0    
                    config = Configuration_BCI.Configuration_BCI('BCI_Framework', dataset)
                    complete_jobs = np.zeros(config.configuration['number_of_subjects'])
                    
                    all_mvs_have_same_lebngth = all(map(lambda x: x == config.configuration['movement_trial_size_list'][0], config.configuration['movement_trial_size_list']))
                    
                    
                    for subj_ind, subj in  enumerate(config.configuration['subject_names_str']):
                        
                        if all_mvs_have_same_lebngth:
                            all_subjects_candidates_list = all_subjects_candidates_dict[(dataset_ind, optimization_type)][0]
                        else:
                            all_subjects_candidates_list = all_subjects_candidates_dict[(dataset_ind, optimization_type)][subj_ind]
                    
                        sp = spearmint_lite.spearmint_lite(Job_Params, all_subjects_candidates_list, config, optimization_type)
                
                        finished[(subj+str(dataset_ind), optimization_type)] = sp.main(Job_Params, complete_jobs, subj)
                        
                    sleep(1)
        
        first_iteration = False