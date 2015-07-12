import os
import sys
os.chdir('../bci_framework')
sys.path.append('./BCI_Framework')

import Configuration_BCI
import Main
import Single_Job_runner as SJR
import numpy as np


if __name__ == '__main__':

    feature_list = ['BP', 'AR', 'logbp', 'morlet', 'wackerman']
    dataset_name = 'BCICIII3b'
    myPython_path = 'python'
    dir = 'BCI_Framework'
    config = Configuration_BCI.Configuration_BCI(dir, dataset_name)

    list_of_classifiers = config.configuration['durations_dict'].keys()
    for subject in config.configuration['subject_names_str']:
        
        for feature in feature_list:
            minimum = np.inf

            for classifier_name in list_of_classifiers:
            
                bcic = Main.Main( dir, dataset_name, classifier_name, feature, 'ALL', -1, myPython_path)
                res_dir = os.path.join(config.configuration['results_path_str'], classifier_name + '/' + feature)
#                 if os.path.exists(res_dir):
                    
                best_params, aa , best_score = bcic.find_learners_optimal_params(subject)
                if best_score < minimum:
                    minimum = best_score
                    minimum_params = aa
                    minimum_classifer = classifier_name
            print subject, minimum_classifer, feature, best_score, aa
        
    
    
