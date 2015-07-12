import sys
sys.path.append('./BCI_Framework')
import Configuration_BCI
import os


if __name__ == '__main__':

    classifiers_dict = {'Boosting':0, 'LogisticRegression':1, 'RANDOM_FOREST':2,'SVM':3, 'LDA':4, 'QDA':5 , 'MLP':6}
    features_dict = {'BP':0, 'logbp':1, 'morlet':2, 'AR':3}
    
    config = Configuration_BCI.Configuration_BCI("BCI_Framework", 'SM2', 'CSP')
    
    for classifier in classifiers_dict.keys():
        for feature in features_dict.keys():
            
            print classifier +'-' + feature + '-' + str(len(os.listdir(config.configuration['results_path_str'] + '/' + classifier + '/' + feature)))