from sklearn.ensemble import RandomForestClassifier
import RandomForest_BCI
import SVM_Learner
import os
import sys
import Configuration_BCI
import numpy as np
import Data_Preprocessor as Data_Proc
import Error
import logging 
import GLM_Learner
import Boosting_learner
import GDA_Learner
# import MLP_Learner
from numpy.core.defchararray import isdigit
import Learner_Manager
import re

class Simple_Job_Runner:

    def __init__(self, dir, learner_name, feature_extractor_name, dataset_name, optimization_type, bo_type):#, channels, number_of_csps):
        """  """
        self.logging = logging
        self.optimization_type = optimization_type
        self.bo_type = bo_type
#         number_of_csps = int(number_of_csps)
        self.config = Configuration_BCI.Configuration_BCI(dir, dataset_name)
#         self.config.set_numberof_channels(number_of_csps)

        if self.config.configuration['logging_level_str'] == 'INFO':
            self.logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
        else:
            self.logging.basicConfig(level=logging.NOTSET)
            
        self.logging.info('begin creating Simple_Job_Runner')

        self.results_path = self.config.configuration['results_path_str']            
        self.results_opt_path = self.config.configuration['results_opt_path_str']
            
        self.results_path, self.results_opt_path = Single_Job_Runner.set_results_path(self.results_path, self.results_opt_path, learner_name, feature_extractor_name, self.optimization_type, self.bo_type)
        self.feature_matrix_path = os.path.join(self.config.configuration['feature_matrix_dir_name_str'], feature_extractor_name)
        
        self.my_Learner_Manager = Learner_Manager.Learner_Manager(self.config, learner_name, feature_extractor_name)
        
        self.logging.info('Simple_Job_Runner instance created, learning algorithm is: %s resutls_path is: %s results_opt_path is: %s'
                          , learner_name, self.results_path, self.results_opt_path)

    def set_channel_parameters(self, channel_type):
        
        temp = re.findall(r'-?\d+', channel_type)
        if len(temp) == 0:
            self.number_of_csps = -1
        else:
            self.number_of_csps = temp[0]
            
        self.channel_type = re.findall(r'[A-Z]+', channel_type)
        if len(self.channel_type) != 0:
            self.channel_type = self.channel_type[0]
        
        if channel_type == []:
            self.channel_type = "ALL"
            self.number_of_csps = -1 
        
        self.config.set_channel_type(self.channel_type, self.number_of_csps)

    @staticmethod
    def create_params_list(params_dict):  
        
        if "cutoff_frequencies_low_list" in params_dict:
            params_list = [float(params_dict['discard_mv_begin']), float(params_dict['discard_mv_end']), float(params_dict['discard_nc_begin']),
                       float(params_dict['discard_nc_end']), float(params_dict['window_size']), float(params_dict['window_overlap_size']), 
                           params_dict['cutoff_frequencies_low_list'], params_dict['cutoff_frequencies_high_list'], params_dict['channel_type']]
        
        else:
            params_list = [float(params_dict['discard_mv_begin']), float(params_dict['discard_mv_end']), float(params_dict['discard_nc_begin']),
                       float(params_dict['discard_nc_end']), float(params_dict['window_size']), float(params_dict['window_overlap_size']), params_dict['channel_type']]
            
        return params_list
    
    @staticmethod
    def check_if_job_exists(resutls_path, subject, params_dict):
        """ """
        
        params_list = Simple_Job_Runner.create_params_list(params_dict)
                
        out_name = Simple_Job_Runner.generate_learner_output_file_name(params_list, subject)
        fname = os.path.join(resutls_path, out_name)
        if not os.path.isfile(fname):
            return False
        else:
            return True
        
    @staticmethod
    def generate_learner_output_file_name(params_list, subject):
        
        out_file_name = str(params_list[0])
        
        for param in params_list[1:]:
            out_file_name += '_' + str(param)
        out_file_name += '_' + subject
        
        return out_file_name   

    @staticmethod
    def set_results_path( results_path, opt_results_path, learner_name, feature_extractor_name, optimization_type = "", BO_selection_type = ""):
        """set the results paths for both the cross validation and optimal learner"""
        if optimization_type == "":
            results_path = os.path.join( results_path, learner_name, feature_extractor_name)
            opt_results_path = os.path.join( opt_results_path, learner_name, feature_extractor_name)
        else:
            results_path = os.path.join( results_path, learner_name, feature_extractor_name, str(optimization_type), BO_selection_type)
            opt_results_path = os.path.join( opt_results_path, learner_name, feature_extractor_name, str(optimization_type), BO_selection_type)            
        
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        
        if not os.path.exists(opt_results_path):
            os.makedirs(opt_results_path)
        
        return results_path, opt_results_path

    @staticmethod
    def set_feature_matrix_path( feature_matrix_path, feature_extractor_name):
        
        feat_path = os.path.join( feature_matrix_path, feature_extractor_name)
    
        if not os.path.exists(feat_path):
            os.makedirs(feat_path)
        
        return feat_path

    
    
    def set_subject(self, subject):
        self.subject = subject
    
    def set_params_dict(self, params_dict):
  
          
        self.params_dict = dict(params_dict)      
        self.params_list = Simple_Job_Runner.create_params_list(params_dict)
        self.discard_mv_begin = float(params_dict.pop('discard_mv_begin'))
        self.discard_mv_end = float(params_dict.pop('discard_mv_end'))
        self.discard_nc_begin = float(params_dict.pop('discard_nc_begin'))
        self.discard_nc_end = float(params_dict.pop('discard_nc_end'))
        self.window_size = float(params_dict.pop('window_size'))
        self.window_overlap_size = float(params_dict.pop('window_overlap_size'))
        
        channel_type = params_dict.pop('channel_type')
        
        if "cutoff_frequencies_low_list" in params_dict:
            self.cutoff_frequencies_low_list = params_dict.pop('cutoff_frequencies_low_list')
            self.cutoff_frequencies_high_list = params_dict.pop('cutoff_frequencies_high_list')
            #self.params_list = [self.discard_mv_begin, self.discard_mv_end, self.discard_nc_begin, 
            #               self.discard_nc_end, self.window_size, self.window_overlap_size, 
            #               self.cutoff_frequencies_low_list, self.cutoff_frequencies_high_list, channel_type]
        
        else:
            self.cutoff_frequencies_low_list = None
            self.cutoff_frequencies_high_list = None
            #self.params_list = [self.discard_mv_begin, self.discard_mv_end, self.discard_nc_begin,
            #           self.discard_nc_end, self.window_size, self.window_overlap_size, channel_type]
        
        self.set_channel_parameters(channel_type)
        self.classfier_params = params_dict
    
    def set_out_file_name(self):

        out_file_name = Single_Job_Runner.generate_learner_output_file_name(self.params_list, self.subject)
        
        self.my_Learner_Manager.set_output_file_path(os.path.join(self.results_path, out_file_name))
        self.my_Learner_Manager.set_output_file_opt_path(os.path.join(self.results_opt_path, out_file_name))
        self.my_Learner_Manager.set_output_file_opt_classifier(os.path.join(self.results_opt_path, out_file_name + '_classifier'))

class Single_Job_Runner(Simple_Job_Runner):
    
    def __init__(self, dir, learner_name, feature_extractor_name, dataset_name, subject, bo_type, optimization_type, params_dict, OPTIMAL = False):
        """"""
        self.bo_type = bo_type
        self.optimization_type = optimization_type
        self.OPTIMAL = OPTIMAL
        self.set_subject(subject)
        Simple_Job_Runner.__init__(self, dir, learner_name, feature_extractor_name, dataset_name, bo_type, optimization_type)
        self.logging.info('begin creating Single_Job_Runner')

        self.set_params_dict(params_dict)

        #if chain length exists in the parameters dictionary it means that we are going to apply a dynamic classifier to the data
        if self.params_dict.has_key('chain_length'):
            classifier_type = 'dynamic'
        else:
            classifier_type = 'static'

        self.dp = Data_Proc.Data_Preprocessor(self.config, subject, feature_extractor_name, int(self.number_of_csps))

        self.set_out_file_name()    
        self.logging.info('Single_Job_Runner instance created, learning algorithm is: %s resutls_path is: %s results_opt_path is: %s'
                          + ' channel_type %s number of CSP filters: %s subject name is: %s discard_mv_begin is: %s discard_mv_end is: %s'
                          +  ' discard_nc_begin is: %s discard_nc_end is: %s window_size is: %s window_overlap_size is: %s and out_file_name is: %s' ,
                            learner_name, self.results_path, self.results_opt_path, str(self.channel_type), str(self.number_of_csps), self.subject,
                            self.discard_mv_begin, self.discard_mv_end, self.discard_nc_begin, self.discard_nc_end, self.window_size, 
                            self.window_overlap_size, self.my_Learner_Manager.result_path)

    
    def run_learner(self):
        """This function receives the parameters for generating the dataset and then runs the learner """
        
        self.logging.info('begin running the learning algorithm!')
        params_list = Simple_Job_Runner.create_params_list(self.params_dict)
        #[ self.discard_mv_begin, self.discard_mv_end, self.discard_nc_begin, self.discard_nc_end, self.window_size, self.window_overlap_size ]
        self.logging.info('list of parameters given to the learner: %s', ' '.join(map(str,params_list)))

        #if features have already been extracted
        if self.check_if_feature_file_exist(): 
            Xs, Y, Xs_test, Y_test = self.load_features()
        else: #otherwise extract features
            if self.cutoff_frequencies_low_list is None:
                Xs, Y = self.dp.extract_data_samples_forall_feature_params(self.subject, self.params_dict, False)
                Xs_test, Y_test = self.dp.extract_data_samples_forall_feature_params(self.subject, self.params_dict, True)
            else:
                Xs, Y = self.dp.extract_data_samples_forall_feature_params(self.subject, self.params_dict, False, cutoff_frequencies_low_list = self.cutoff_frequencies_low_list, cutoff_frequencies_high_list = self.cutoff_frequencies_high_list)
                Xs_test, Y_test = self.dp.extract_data_samples_forall_feature_params(self.subject, self.params_dict, True, cutoff_frequencies_low_list = self.cutoff_frequencies_low_list, cutoff_frequencies_high_list = self.cutoff_frequencies_high_list)
        
            if self.config.configuration["save_feature_matrix"] == 1:
                self.save_features(X_train=Xs, Y_train=Y, X_test=Xs_test, Y_test= Y_test)
            
        self.logging.info('finished features extraction from the training and test data')
        
        if self.OPTIMAL:
            self.logging.info('begin training optimal classifier')
            self.my_Learner_Manager.train_learner(Xs, Y, Xs_test, Y_test,self.classfier_params, optimal = self.OPTIMAL)
        else :
            self.logging.info('begin training classifier')
            self.my_Learner_Manager.train_learner(Xs, Y, optimal = self.OPTIMAL)

    def check_if_feature_file_exist(self):
        
        # params_list = Simple_Job_Runner.create_params_list(self.params_dict)

        out_name = Simple_Job_Runner.generate_learner_output_file_name(self.params_list, self.subject)
        out_path = os.path.join(self.feature_matrix_path, out_name + '.npz')
        
        if not os.path.isfile(out_path):
            return False
        else:
            return True
        
    def save_features(self, X_train = [], Y_train = [], X_test = [], Y_test = []):
        """save feature matrices to file"""
        out_name = Simple_Job_Runner.generate_learner_output_file_name(self.params_list, self.subject)
        out_path = os.path.join(self.feature_matrix_path, out_name)
        
        np.savez_compressed(out_path, X_train, Y_train, X_test, Y_test)
        self.logging.info('saved features to file!')
        
    def load_features(self):
        """If feature matrices are already saved to files, load them from file"""
        out_name = Simple_Job_Runner.generate_learner_output_file_name(self.params_list, self.subject)
        out_path = os.path.join(self.feature_matrix_path, out_name + '.npz')
        
        A = np.load(out_path)
        
        self.logging.info('load features from file!')
        return A['arr_0'], A['arr_1'], A['arr_2'], A['arr_3']
        
    # def save_feature_matrices(self):
    #     """This function receives the parameters for generating the dataset and then runs the learner """
    #
    #     self.logging.info('Writing feature matrices to file!')
    #     params_list = [ self.discard_mv_begin, self.discard_mv_end, self.discard_nc_begin, self.discard_nc_end,
    #                    self.window_size, self.window_overlap_size, self.params_dict['channel_type']]
    #     self.logging.info('list of parameters given to the learner: %s', ' '.join(map(str,params_list)))
    #
    #     if self.cutoff_frequencies_low_list is None:
    #         Xs, Y = self.dp.extract_data_samples_forall_feature_params(self.subject, self.params_dict, False)
    #         Xs_test, Y_test = self.dp.extract_data_samples_forall_feature_params(self.subject, self.params_dict, True)
    #     else:
    #         Xs, Y = self.dp.extract_data_samples_forall_feature_params(self.subject, self.params_dict, False, cutoff_frequencies_low_list = self.cutoff_frequencies_low_list, cutoff_frequencies_high_list = self.cutoff_frequencies_high_list)
    #         Xs_test, Y_test = self.dp.extract_data_samples_forall_feature_params(self.subject, self.params_dict, True, cutoff_frequencies_low_list = self.cutoff_frequencies_low_list, cutoff_frequencies_high_list = self.cutoff_frequencies_high_list)
    #
    #     out_name = Simple_Job_Runner.generate_learner_output_file_name(params_list, self.subject)
    #     out_path = os.path.join(self.feature_matrix_path, out_name)
    #
    #     ###########################################
    #     np.savez(out_path, X_train=Xs[0], Y_train=Y, Y_test=Y_test, X_test= Xs_test)
    #     self.logging.info('feature extraction finished!')

        
    
    def calc_correlation(self, Xs):
        
        for X in Xs:
            pass
            print np.corrcoef(X.T)
#             np.savetxt("correlations.csv",  np.corrcoef(X.T), delimiter=",")
            print 3
    
    @staticmethod
    def check_if_job_exists(resutls_path, subject, params_dict):
        """ """
        params_list = Simple_Job_Runner.create_params_list(params_dict)

        #if "cutoff_frequencies_low_list" in params_dict:
        #    params_list = [float(params_dict['discard_mv_begin']), float(params_dict['discard_mv_end']), float(params_dict['discard_nc_begin']),
        #               float(params_dict['discard_nc_end']), float(params_dict['window_size']), float(params_dict['window_overlap_size']), params_dict['cutoff_frequencies_low_list'],
        #               params_dict['cutoff_frequencies_high_list'], params_dict['channel_type']]
        
        #else:
        #    params_list = [float(params_dict['discard_mv_begin']), float(params_dict['discard_mv_end']), float(params_dict['discard_nc_begin']),
        #               float(params_dict['discard_nc_end']), float(params_dict['window_size']), float(params_dict['window_overlap_size']), params_dict['channel_type']]
        
        out_name = Simple_Job_Runner.generate_learner_output_file_name(params_list, subject)
        fname = os.path.join(resutls_path, out_name)
        if not os.path.isfile(fname):
            return False
        else:
            return True
    
if __name__ == '__main__': # TODO: fix this

    print sys.argv
    def isFloat(string):
        try:
            float(string)
            return True
        except ValueError:
            return False
    
    dir = sys.argv[1]
    learner_name = sys.argv[2]
    feature_extractor_name = sys.argv[3]
    dataset_name = sys.argv[4]
    subject = sys.argv[5]
    optimization_type = sys.argv[6]
    bo_type = sys.argv[7]
    if sys.argv[8] == 'True':
        optimal = True
    else:
        optimal = False
    
    
    params_list = dict([arg.split('=') for arg in sys.argv[9:]])
    params_dict = {}
    for k, v in params_list.items():
        if (k == 'cutoff_frequencies_low_list' or  k == 'cutoff_frequencies_high_list'):
            params_dict[k] = v
        
        elif k == 'fe_params' and ',' in v:
            params_dict[k]  = [int(e) for e in v.split(',')]
        elif isFloat(v):
            params_dict[k] = float(v)
        elif v == 'None':
            params_dict[k] = None
        else:
            params_dict[k] = v
    
    
    job_runner = Single_Job_Runner(dir, learner_name, feature_extractor_name, dataset_name, subject, optimization_type, bo_type, params_dict, optimal)
        
    job_runner.run_learner()
