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

    mypath = '../Candidates'
    chooser_modules = ["GPEIOptChooser", "GPEIOptChooser1", "GPEIOptChooser2", "GPEIOptChooser3", "GPEIOptChooser4", "RandomForestEIChooser", "RandomForestEIChooser1", "RandomForestEIChooser2", "RandomForestEIChooser3", "RandomForestEIChooser4", "RandomChooser","RandomChooser1", "RandomChooser2", "RandomChooser3", "RandomChooser4"]
    datasets = ['BCICIII3b', 'BCICIV2b', 'BCICIV2a']
    optimization_types_dict = {('BCICIII3b','BP'):[2], ('BCICIII3b','morlet'):[1], ('BCICIV2b','BP') : [2], ('BCICIV2b','morlet') : [1], ('BCICIV2a','BP') : [4], ('BCICIV2a','morlet') : [3]}
    features = ['BP']
    
    for dataset in datasets:
        config = Configuration_BCI.Configuration_BCI('BCI_Framework', dataset)
        subjects = config.configuration['subject_names_str']
        for feature in features:
            for chooser_module in chooser_modules:
                for subject in subjects:
                    file_name = "results_" + str(optimization_types_dict[(dataset,feature)][0]) + '_' + str(chooser_module) + '_LogisticRegression_' + feature + '.dat_' + subject
#                     print file_name
                    with open(os.path.join(mypath, file_name)) as cand_file:
                        all_candidates = cand_file.readlines()
                        if len(all_candidates) < 40:
                            print len(all_candidates), file_name
    
    
    