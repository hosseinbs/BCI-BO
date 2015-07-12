import numpy as np
import os
from os import listdir
from os.path import isfile, join
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

    mypath = "../Candidates"
    dataset_subjects_dict = {'O3':'BCICIII3b', 'S4':'BCICIII3b', 'X11':'BCICIII3b', '100':'BCICIV2b', '200':'BCICIV2b', '300':'BCICIV2b','400':'BCICIV2b',
                             '500':'BCICIV2b', '600':'BCICIV2b', '700':'BCICIV2b', '800':'BCICIV2b', '900':'BCICIV2b', '1':'BCICIV2a', '2':'BCICIV2a', 
                             '3':'BCICIV2a','4':'BCICIV2a', '5':'BCICIV2a', '6':'BCICIV2a', '7':'BCICIV2a', '8':'BCICIV2a', '9':'BCICIV2a'}
    onlyfiles = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]

    for f in onlyfiles:
        if '.dat' in f:
            print os.path.join(mypath, f)
            print f
            subject = f.split('_')[-1]
            print subject
            feature = (f.split('_')[-2]).split('.')[0]
            print feature
            optimization_type = int(float(f.split('_')[1]))
            print optimization_type
            chooser_module = f.split('_')[2]
            print chooser_module
            
            with open(os.path.join(mypath, f)) as candidates_file:
                all_candidates = candidates_file.readlines()
                for current_candidate in all_candidates:
                    last_candidate = current_candidate.split()
                    if last_candidate[0] == 'P':
                        params = map(float, last_candidate[2:])
    
#                         print "Bayesian Optimization for BCI"
                        dataset = dataset_subjects_dict[subject]#'BCICIII3b'
                        classifier = 'LogisticRegression'
                        
                        first_iteration = True
                        finished = {}
                                
                        class Job_Params:
                            job_dir = '../Candidates'
                            num_all_jobs = 40
                            dataset = dataset
                            seed = 1
                            classifier_name = classifier
                            feature_extraction = feature
                            n_concurrent_jobs = 1
                            chooser_module = chooser_module
                            n_initial_candidates = 0
                            
                        all_subjects_candidates_list = BO_BCI.generate_all_candidates(dataset, optimization_type)
                        config = Configuration_BCI.Configuration_BCI('BCI_Framework', dataset)
                        sp = spearmint_lite.spearmint_lite(Job_Params, all_subjects_candidates_list, config, optimization_type)
                            
#                         print params
                        sp.run_job(params, subject)                        