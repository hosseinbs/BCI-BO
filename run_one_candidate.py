import spearmint_lite
import os
import sys
os.chdir('../bci_framework')
sys.path.append('./BCI_Framework')

import Main
import Configuration_BCI
import numpy as np

import spearmint_lite

class Job_Params:
    job_dir = 'BCI_Framework'
    num_all_jobs = 100
    dataset = 'BCICIV2b'
    seed = 1
    classifier_name = 'LogisticRegression'
    feature_extraction = 'BP'
    n_concurrent_jobs = 3
    chooser_module = "GPEIOptChooser"
    n_initial_candidates = 12


if __name__ == '__main__':

    BO_type = 2
    config = Configuration_BCI.Configuration_BCI('BCI_Framework', 'BCICIV2b')
    
    sp = spearmint_lite.spearmint_lite(Job_Params, [], config, BO_type)
    params = [687.0, 1.0, 1.5, 3.5]
    subj = '100'
    sp.run_job(params, subj)