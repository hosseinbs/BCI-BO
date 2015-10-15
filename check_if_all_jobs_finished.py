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
    optimization_types_dict = {('BCICIII3b','BP'):[2], ('BCICIII3b','morlet'):[1], ('BCICIV2b','BP') : [2], ('BCICIV2b','morlet') : [1], ('BCICIV2a','BP') : [4], ('BCICIV2a','morlet') : [3]}
    BO_selection_types = ["GPEIOptChooser1", "RandomForestEIChooser1", "RandomChooser1",
                           "GPEIOptChooser2", "RandomForestEIChooser2", "RandomChooser2",
                           "GPEIOptChooser3", "RandomForestEIChooser3", "RandomChooser3",
                           "GPEIOptChooser4", "RandomForestEIChooser4", "RandomChooser4",
                           "GPEIOptChooser5", "RandomForestEIChooser5", "RandomChooser5",
                            "GPEIOptChooser6", "RandomForestEIChooser6", "RandomChooser6",
                           "GPEIOptChooser7", "RandomForestEIChooser7", "RandomChooser7",
                           "GPEIOptChooser8", "RandomForestEIChooser8", "RandomChooser8",
                           "GPEIOptChooser9", "RandomForestEIChooser9", "RandomChooser9",
                           "GPEIOptChooser10", "RandomForestEIChooser10", "RandomChooser10"]

    datasets = ['BCICIV2b', 'BCICIV2a','BCICIII3b']
    # optimization_types_dict = {('BCICIII3b','BP'):[2], ('BCICIII3b','morlet'):[1], ('BCICIV2b','BP') : [2], ('BCICIV2b','morlet') : [1], ('BCICIV2a','BP') : [4], ('BCICIV2a','morlet') : [3]}
    features = ['morlet', 'BP']
    
    for dataset in datasets:
        config = Configuration_BCI.Configuration_BCI('BCI_Framework', dataset)
        subjects = config.configuration['subject_names_str']
        for feature in features:
            for chooser_module in BO_selection_types:
                for subject in subjects:
                    file_name = "results_" + str(optimization_types_dict[(dataset,feature)][0]) + '_' + str(chooser_module) + '_LogisticRegression_' + feature + '.dat_' + subject
#                     print file_name
                    with open(os.path.join(mypath, file_name)) as cand_file:
                        all_candidates = cand_file.readlines()
                        if len(all_candidates) < 40:
                            print len(all_candidates), file_name
    
    
    