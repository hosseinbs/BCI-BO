import os
import sys
sys.path.append('./BCI_Framework')

import Main
import Single_Job_runner as SJR
#import pylab as pl
import numpy as np


if __name__ == '__main__':

    
    
    if os.name == 'posix':

        if os.uname()[1] == 'bugaboo.westgrid.ca':    
            myPython_path = '/home/hosseinb/data/software/MyPython/usr/local/bin/python'
        elif os.uname()[1] == 'tatanka.local': 
            myPython_path = '/home/hosseinb/bin/python'
        else:
            myPython_path = 'python'
#            print 'set python pass!!!!!!'
#            sys.exit()
    elif os.name == 'nt':
        myPython_path = 'python'
        
               
## channels = 'Cs' or 'ALL'
    channels = 'ALL'     
    number_of_CSPs = -1
#
### instantiate a Main class object 
    bciciv2b = Main.Main('BCI_Framework','BCICIV2b','SVM_linear', 'wackerman', channels, number_of_CSPs, myPython_path)
#
### perform grid search to find optimal parameters    
    bciciv2b.run_learner_gridsearch()
#
### find best classifier and apply it on test data
#    bciciv2b.test_learner()
#   
### instantiate a Main class object 
    bciciv2b = Main.Main('BCI_Framework','BCICIV2b','SVM_rbf', 'wackerman', channels, number_of_CSPs, myPython_path)
#
### perform grid search to find optimal parameters    
    bciciv2b.run_learner_gridsearch()
#
### find best classifier and apply it on test data
#    bciciv2b.test_learner()
#
### instantiate a Main class object 
    bciciv2b = Main.Main('BCI_Framework','BCICIV2b','RANDOM_FOREST', 'wackerman', channels, number_of_CSPs, myPython_path)
#
### perform grid search to find optimal parameters    
    bciciv2b.run_learner_gridsearch()

## find best classifier and apply it on test data
#    bciciv2b.test_learner()

## instantiate a Main class object 
    bciciv2b = Main.Main('BCI_Framework','BCICIV2b','LogisticRegression_l1', 'wackerman', channels, number_of_CSPs, myPython_path)
#
### perform grid search to find optimal parameters    
    bciciv2b.run_learner_gridsearch()
#
### find best classifier and apply it on test data
#    bciciv2b.test_learner()
### instantiate a Main class object 
    bciciv2b = Main.Main('BCI_Framework','BCICIV2b','LogisticRegression_l2', 'wackerman', channels, number_of_CSPs, myPython_path)

### perform grid search to find optimal parameters    
    bciciv2b.run_learner_gridsearch()

## find best classifier and apply it on test data
#    bciciv2b.test_learner()
## instantiate a Main class object 
    bciciv2b = Main.Main('BCI_Framework','BCICIV2b','Boosting', 'wackerman', channels, number_of_CSPs, myPython_path)
#
### perform grid search to find optimal parameters    
    bciciv2b.run_learner_gridsearch()

## find best classifier and apply it on test data
#    bciciv2b.test_learner()