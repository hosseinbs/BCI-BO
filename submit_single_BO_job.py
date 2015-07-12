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

import BO_BCI    

if __name__ == '__main__':

    print "Bayesian Optimization for BCI"
#     dataset = 'BCICIII3b'
    dataset = 'BCICIV2a'
    classifier = 'LogisticRegression'
    feature = 'morlet'
    optimization_type = 3
    
    first_iteration = True
    finished = {}
            
    class Job_Params:
        job_dir = 'BCI_Framework'
        num_all_jobs = 40
        dataset = dataset
        seed = 1
        classifier_name = classifier
        feature_extraction = feature
        n_concurrent_jobs = 1
        chooser_module = "GPEIOptChooser1"
        n_initial_candidates = 0
        
    all_subjects_candidates_list = BO_BCI.generate_all_candidates(dataset, optimization_type)
            
    config = Configuration_BCI.Configuration_BCI('BCI_Framework', dataset)
    
    sp = spearmint_lite.spearmint_lite(Job_Params, all_subjects_candidates_list, config, optimization_type)
        
#     params = [93.0, 64.0]
#     sp.run_job(params, 'X11')
# #     
#     params = [125.0, 0.0, 3.0]
#     sp.run_job(params, '2')
#     
#     params = [0.0, 125.0, 3.0]
#     sp.run_job(params, '3')
#     
#     params = [62.0, 1.0, 3.0]
#     sp.run_job(params, '4')
#      
#     params = [375.0, 0.0, 0.0]
#     sp.run_job(params, '5')
#     
#     params = [0.0, 438.0, 2.0]
#     sp.run_job(params, '6')
#     
#     params = [0.0, 313.0, 1.0]
#     sp.run_job(params, '7')
#     
    params = [0.0, 125.0, 3.0]

    [train_val, temp_dur, learner_params] = sp.check_job_complete(params, '7', config)
    job_submitted = sp.run_optimal_job( map(float, params), '7', learner_params)
    
    
    
    
#     sp.run_job(params, '7')
#     
#     params = [0.0, 0.0, 3.0]
#     sp.run_job(params, '9')

#     params = [687.0, 1.0]
#     sp.run_job(params, '300')
    
#     params = [62.0, 126.0]
#     sp.run_job(params, '800')
    
#     params = [62.0, 188.0]
#     sp.run_job(params, '900')