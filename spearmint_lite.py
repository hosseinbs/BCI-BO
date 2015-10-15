#
# This code is adapted from jasper Snoek's Bayesian optimization code
#
##
# Copyright (C) 2012 Jasper Snoek, Hugo Larochelle and Ryan P. Adams
#                                                                                                                                                                              
# This code is written for research and educational purposes only to
# supplement the paper entitled "Practical Bayesian Optimization of
# Machine Learning Algorithms" by Snoek, Larochelle and Adams Advances
# in Neural Information Processing Systems, 2012
#                                                                                                                                                       
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#                                                                                                                                                                       
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#                                                                                                                                                                       
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see
# <http://www.gnu.org/licenses/>.
import optparse
import tempfile
import datetime
import subprocess
import time
import imp
import os
import re
import collections
import time
import numpy as np
import sys
import logging

from ExperimentGrid  import *
#from test.test_coercion import candidates
try: import simplejson as json
except ImportError: import json


sys.path.append('./BCI_Framework')
import Main as Main_BCI
import Configuration_BCI
import Single_Job_runner as SJR

# import win32com.client
#
# There are two things going on here.  There are "experiments", which are
# large-scale things that live in a directory and in this case correspond
# to the task of minimizing a complicated function.  These experiments
# contain "jobs" which are individual function evaluations.  The set of
# all possible jobs, regardless of whether they have been run or not, is
# the "grid".  This grid is managed by an instance of the class
# ExperimentGrid.s
#
# The spearmint.py script can run in two modes, which reflect experiments
# vs jobs.  When run with the --wrapper argument, it will try to run a
# single job.  This is not meant to be run by hand, but is intended to be
# run by a job queueing system.  Without this argument, it runs in its main
# controller mode, which determines the jobs that should be executed and
# submits them to the queueing system.
#

class spearmint_lite:

    def __init__(self, job_params, candidates_list, config, bo_type):

        if job_params == None and candidates_list == None:
            self.type = bo_type
            self.config = config

        else:
            self.type = bo_type
            self.config = config
            self.logging = logging
            if self.config.configuration['logging_level_str'] == 'INFO':
                self.logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
            else:
                self.logging.basicConfig(level=logging.NOTSET)
        
            self.logging.info('started building spearmint_lite!')
        
            self.myPython_path = 'python'
            
            self.job_dir = job_params.job_dir
            self.num_jobs = job_params.num_all_jobs
            self.dataset = job_params.dataset
            self.seed = job_params.seed
            self.classifier_name = job_params.classifier_name
            self.feature_extraction = job_params.feature_extraction
            self.n_concurrent_jobs = job_params.n_concurrent_jobs
            self.n_initial_candidates_length = job_params.n_initial_candidates

            self.chooser_module = job_params.chooser_module
        
            self.candidates = np.array(candidates_list)
            self.job_results_file = 'results_' + str(self.type) + '_' + job_params.chooser_module + '_'+ self.classifier_name + '_' + self.feature_extraction + '.dat'

            chooser_module_passed_to_framework = ''.join([i for i in job_params.chooser_module if not i.isdigit()])
            self.bcic = Main_BCI.Main('BCI_Framework', self.dataset, self.classifier_name, self.feature_extraction, self.type, chooser_module_passed_to_framework, 'ALL', -1)
            self.SJR = SJR.Simple_Job_Runner(self.bcic.dir, self.bcic.learner_name, self.bcic.feature_extractor_name,self.bcic.dataset_name, bo_type, chooser_module_passed_to_framework)
        
            self.finished = False
        
            self.logging.info('An spearmint instance has been built! num_jobs: %s dataset: %s random seed: %s classifier: %s feature_extractor: %s'
                          + 'number of concurrent jobs: %s  results_file: %s', str(self.num_jobs), str(self.dataset),
                          str(self.seed), self.classifier_name, self.feature_extraction, str(self.n_concurrent_jobs), self.job_results_file)
    
    def main(self, Job_Params, complete_jobs, subj):
        
        
        args = [self.job_dir]
        expt_dir  = os.path.realpath(args[0])
        work_dir  = os.path.realpath('.')
        expt_name = os.path.basename(expt_dir)
    
        if not os.path.exists(expt_dir):
            sys.stderr.write("Cannot find experiment directory '%s'.  Aborting.\n" % (expt_dir))
            sys.exit(-1)
    
        self.logging.info('optimizaing parameters for %s', subj)
        # Create the experimental grid
        
        # Load up the chooser module.
        if "RandomChooser" in self.chooser_module:
            module  = __import__("RandomChooser")
        if "RandomForestEIChooser" in self.chooser_module:
            module  = __import__("RandomForestEIChooser")
        elif "GPEIOptChooser" in self.chooser_module:
            module  = __import__("GPEIOptChooser")
        
        chooser = module.init(expt_dir, "", subj + '_' + str(self.type) + '_' + Job_Params.chooser_module + '_'+ self.classifier_name + '_' + self.feature_extraction)#, gmap, self)
    
#         self.run_initial_candidates(subj)
        res_file, values, complete, duarations, pendings = self.read_update_results(self.job_dir, subj)#, options, gmap)

        complete_jobs = len(complete)
        
        temp = self.num_jobs 
        if (len(complete) + len(pendings)) < self.num_jobs and len(pendings) < self.n_concurrent_jobs:
            
            self.n_concurrent_jobs = self.n_concurrent_jobs - len(pendings)        
            self.main_controller(args, subj, chooser)
        
        elif len(complete) >= self.num_jobs :
            self.finished = True
            # self.prepare_opt_results_file(subj)
    
        self.num_jobs = temp
            
        return self.finished
    ##############################################################################
    ##############################################################################
    
    def main_controller(self, args, subj, chooser):
    
        
        # Read in parameters and values observed so far
        for i in xrange(0,self.n_concurrent_jobs):
    
            res_file, values, complete, durations, pending = self.read_results(self.job_dir, subj) 
    
            # Some stats
            sys.stderr.write("#Complete: %d #Pending: %d\n" % 
                             (complete.shape[0], pending.shape[0]))
    
            # Let's print out the best value so far
            if type(values) is not float and len(values) > 0:
                best_val = np.min(values)
                best_job = np.argmin(values)
                sys.stderr.write("Current best: %f (job %d)\n" % (best_val, best_job))
        
            # Now lets get the next job to run
            # First throw out a set of candidates on the unit hypercube
            # Increment by the number of observed so we don't take the
            # same values twice
            off = pending.shape[0] + complete.shape[0]
            
            # Ask the chooser to actually pick one.
            # First mash the data into a format that matches that of the other
            # spearmint drivers to pass to the chooser modules.
            self.delete_extra_candidates(np.copy(complete)) # deleting completed candidates
            self.delete_extra_candidates(np.copy(pending)) # deleting pending candidates
                               
            grid = np.copy(self.candidates)
    
            if (complete.shape[0] > 0):
                grid = np.vstack((complete, grid))
            if (pending.shape[0] > 0):
                grid = np.vstack((grid, pending))
            
#             grid = np.asarray(self.candidates)
            grid_idx = np.hstack((np.zeros(complete.shape[0]),
                                  np.ones(self.candidates.shape[0]),
                                  1.+np.ones(pending.shape[0])))
            
            grid_scaled_in_unit_cube = grid / np.max(grid, axis = 0)
            job_id = chooser.next(grid_scaled_in_unit_cube, np.squeeze(values), durations, np.nonzero(grid_idx == 1)[0], np.nonzero(grid_idx == 2)[0], np.nonzero(grid_idx == 0)[0])
            
            # If the job_id is a tuple, then the chooser picked a new job not from
            # the candidate list
            if isinstance(job_id, tuple):
                (job_id, candidate) = job_id
            else:
                candidate = grid[job_id,:]
    
            sys.stderr.write("Selected job %d from the grid.\n" % (job_id))
            if pending.shape[0] > 0:
                pending = np.vstack((pending, candidate))
            else:
                pending = np.matrix(candidate)
    
            params = candidate[:]

            # Now lets write this candidate to the file as pending
            output = ""
            for p in params:
                output = output + str(p) + " "
                
            output = "P P " + output + "\n"
            outfile = open(res_file,"a")
            outfile.write(output)
            outfile.close()        
            
            sys.path.append( os.path.join('.', self.job_dir) )
    
            self.run_job(params, subj)#, gmap)
            
    
    def delete_extra_candidates(self, to_be_deleted_candidates):
        
        while len(to_be_deleted_candidates) != 0:
            to_be_deleted = to_be_deleted_candidates[0,:]
            to_be_deleted_candidates = np.delete(to_be_deleted_candidates, (0), axis=0)
            
            for candidate_id, candidate in enumerate(self.candidates):
                if all(candidate == to_be_deleted):
                    
                    self.candidates = np.delete(self.candidates, (candidate_id), axis=0)

                    break
                    
    def run_initial_candidates(self, subject):
        
        expt_dir  = os.path.realpath(self.job_dir)

        res_file = os.path.join(expt_dir, self.job_results_file + '_' + subject)
        
        if os.path.exists(res_file):
            with open( res_file, 'r') as candidates_file:
                already_submitted_candidates_raw = candidates_file.readlines()

            #this is to prevent multiple submission of initial candidates         
            if(len(already_submitted_candidates_raw) >= self.n_initial_candidates_length):
                return
        
        for candidate in self.candidates[0:self.n_initial_candidates_length]:
            already_exists = self.run_job(candidate, subject)
            
            if not already_exists:
                output = ""
                for p in candidate:
                    output = output + str(p) + " "
                    
                output = "P P " + output + "\n"
                outfile = open( res_file,"a")
                outfile.write(output)
                outfile.close()        
            
            
    def run_job(self, params, subject):
        """  """
        params_dict = self.generate_params_dict(params, subject)#, gmap)
        chooser_module_passed_to_framework = ''.join([i for i in self.chooser_module if not i.isdigit()])
        main_runner = Main_BCI.Main('BCI_Framework', self.dataset, self.classifier_name, self.feature_extraction, self.type, chooser_module_passed_to_framework)#, channels, -1, self.myPython_path)
        
        return main_runner.run_learner_BO(subject, params_dict)

    def generate_params_dict(self, params, subject):

        channels_list = ['ALL-1', 'CSP2', 'CSP4', 'CSP6', 'CS']
        n_channels = self.config.configuration["number_of_channels"]
        if self.type == 1:
            params_dict = {'discard_mv_begin' : params[0], 'discard_mv_end' : params[1], 
                        'discard_nc_begin' : 0, 'discard_nc_end' : 0, 'window_size' : -1, 'window_overlap_size' : 0,
                        'fe_params':None, 'channel_type': 'ALL-1'}

        elif self.type == 2:
            if params[2] == params[4] and params[3] == params[5]:
                cutoff_frequencies_low_list = [params[2]] 
                cutoff_frequencies_high_list = [params[3]] 
            else:
                cutoff_frequencies_low_list = [params[2]]  + [params[4]] 
                cutoff_frequencies_high_list = [params[3]] + [params[5]] 
            
            cutoff_frequencies_low = '_'.join(str(d) for d in cutoff_frequencies_low_list)
            cutoff_frequencies_high = '_'.join(str(d) for d in cutoff_frequencies_high_list)
        
            params_dict = {'discard_mv_begin' : params[0], 'discard_mv_end' : params[1], 
                        'discard_nc_begin' : 0, 'discard_nc_end' : 0, 'window_size' : -1, 'window_overlap_size' : 0,
                        'cutoff_frequencies_low_list': cutoff_frequencies_low,
                        'cutoff_frequencies_high_list':cutoff_frequencies_high, 'fe_params':None, 'channel_type':'ALL-1'}

        elif self.type == 3:

            params_dict = {'discard_mv_begin' : params[0], 'discard_mv_end' : params[1], 
                        'discard_nc_begin' : 0, 'discard_nc_end' : 0, 'window_size' : -1, 'window_overlap_size' : 0,
                        'fe_params':None, 'channel_type':channels_list[int(float(params[-1]))]}

        elif self.type == 4:
            
            if params[2] == params[4] and params[3] == params[5]:
                cutoff_frequencies_low_list = [params[2]] 
                cutoff_frequencies_high_list = [params[3]] 
            else:
                cutoff_frequencies_low_list = [params[2]] + [params[4]] 
                cutoff_frequencies_high_list = [params[3]] + [params[5]] 
                            
            cutoff_frequencies_low = '_'.join(str(d) for d in cutoff_frequencies_low_list)
            cutoff_frequencies_high = '_'.join(str(d) for d in cutoff_frequencies_high_list)
        
            params_dict = {'discard_mv_begin' : params[0], 'discard_mv_end' : params[1], 
                        'discard_nc_begin' : 0, 'discard_nc_end' : 0, 'window_size' : -1, 'window_overlap_size' : 0,
                        'cutoff_frequencies_low_list': cutoff_frequencies_low,
                        'cutoff_frequencies_high_list':cutoff_frequencies_high, 'fe_params':None, 'channel_type':channels_list[int(float(params[-1]))]}

        return params_dict
    
    def run_optimal_job(self, params, subject, learner_params):
        """ """
        print subject
        print params
        params_dict = self.generate_params_dict(params, subject)
        params_dict = dict(params_dict.items() + learner_params.items())
        
        chooser_module_passed_to_framework = ''.join([i for i in self.chooser_module if not i.isdigit()])    
        main_runner = Main_BCI.Main('BCI_Framework', self.dataset, self.classifier_name, self.feature_extraction, self.type, chooser_module_passed_to_framework)#, channels, -1, self.myPython_path)
        
        return main_runner.run_optimal_learner_BO(subject, params_dict)

    # And that's it
    def read_results(self, job_dir, subj):
    
        expt_dir  = os.path.realpath(job_dir)
        work_dir  = os.path.realpath('.')
        expt_name = os.path.basename(expt_dir)
        
        values = np.array([])
        complete = np.array([])
        pending = np.array([])
        durations = np.array([])
        
        res_file = os.path.join(expt_dir, self.job_results_file + '_' + subj)
        if not os.path.exists(res_file):
            thefile = open( res_file, 'w')
            thefile.write("")
            thefile.close()
    
        index = 0
    
        infile = open( res_file, 'r')
        for line in infile.readlines():
            # Each line in this file represents an experiment
            # It is whitespace separated and of the form either
            # <Value> <time taken> <space separated list of parameters>
            # indicating a completed experiment or
            # P P <space separated list of parameters>
            # indicating a pending experiment
            expt = line.split()
            if (len(expt) < 3):
                continue
    
            val = expt.pop(0)
            dur = expt.pop(0)
#             variables = gmap.to_unit(expt)
            variables = map(float, expt)
            if val == 'P':
                if pending.shape[0] > 0:
                    pending = np.vstack((pending, variables))
                else:
                    pending = np.array([np.array(variables)])
            else:
                if complete.shape[0] > 0:
                    values = np.vstack((values, float(val)))
                    complete = np.vstack((complete, variables))
                    durations = np.vstack((durations, float(dur)))
                else:
                    values = float(val)
                    complete = np.array([np.array(variables)])
                    durations = float(dur)
                
        infile.close()
        
        return res_file , values, complete, durations, pending
    
#     
#     def add_complete_jobs(self, subj, Job_Params, complete,pendings, config, values, res_file, gmap):
#         
#         remaining_pending_indices = len(pendings) * [True]
#         for p_ind, p in enumerate(pendings):
#             
#             p_list = gmap.unit_to_list(np.array(p).squeeze())
#             subj_file_name = '_'.join(map(str,p_list)) + '_' + subj
#             res_dir = config.configuration['results_path_str']
#             if Job_Params.learner == 'RANDOM_FOREST':
#                 res_dir = os.path.join(res_dir, 'RF')
#                 learner = RandomForest_BCI.RandomForest_BCI()
#             elif Job_Params.learner == 'DROPOUT':
#                 res_dir = os.path.join( res_dir, 'DON')
#                 learner = RandomForest_BCI.RandomForest_BCI()
#                 
#             res_file_name = os.path.join(res_dir,subj_file_name)
#             if os.path.isfile(res_file_name):
#                 
#                 remaining_pending_indices[p_ind] = False
#                 val = learner.find_cv_error(res_file_name)
#                 if len(values) == 0:
#                     values = np.array([float(val)])
#                     complete =  p
#                 else:
#                     values = np.vstack((values, float(val)))
#                     complete = np.vstack((complete, p))
#                     
#                 
#         pendings = pendings[remaining_pending_indices]
#         with open(res_file,'w') as outfile:
#             
#             # Now lets write this candidate to the file as pending
#             for c_ind, c in enumerate(complete):
#                 output = ' '.join(map(str,c))
#                 output = str(values[c_ind]) + ' 100 ' + output 
#                 outfile.write(output)
#     
#             for p in pendings:
#                 output = ' '.join(map(str,p))
#                 output = "P P " + output + "\n"
#                 outfile.write(output)
#                 
    
    def read_update_results(self, job_dir, subj):#, options, gmap):
    
        expt_dir  = os.path.realpath(job_dir)
        work_dir  = os.path.realpath('.')
        expt_name = os.path.basename(expt_dir)
        
        values = np.array([])
        complete = np.array([])
        pending = np.array([])
        durations = np.array([])
        
        expts = np.array([])
        expts_vals = []
        expts_durs = []
        
        res_file = os.path.join(expt_dir, self.job_results_file + '_' + subj)
        if not os.path.exists(res_file):
            thefile = open( res_file, 'w')
            thefile.write("")
            thefile.close()
    
        index = 0
    
        with open( res_file, 'r') as infile:
            for line in infile:
                expt = line.split()
                if (len(expt) < 3):
                    continue
        
                val = expt.pop(0)
                dur = expt.pop(0)
    
                [val1, dur1, temp] = self.check_job_complete(expt, subj, self.config)#, gmap)
                if not val1 is None:
                    val, dur = val1, dur1
                
                if expts.shape[0] > 0:
                    expts = np.vstack((expts, expt))
                else:
                    expts = np.matrix(expt)
                expts_vals.append(val)
                expts_durs.append(dur)    
    
                variables = np.array(map(float, expt))
#                 variables = gmap.to_unit(list(expt))
                if val == 'P':
                    if pending.shape[0] > 0:
                        pending = np.vstack((pending, variables))
                    else:
                        pending = np.matrix(variables)
                    
    #                if expts.shape[0] > 0:
    #                    expts_vals = np.vstack((expts_vals, val))
    #                    expts_durs = np.vstack((expts_durs, dur))
    #                else:
    #                    expts_vals = val
    #                    expts_durs = dur
                else:
                    [train_val, temp_dur, learner_params] = self.check_job_complete(expt, subj, self.config)
                    job_submitted = self.run_optimal_job( map(float, expt), subj, learner_params)##########################################################
#                     self.prepare_opt_results_file(expt, subj)###############################################################################################
                    
                    if complete.shape[0] > 0:
                        values = np.vstack((values, float(val)))
                        complete = np.vstack((complete, variables))
                        durations = np.vstack((durations, float(dur)))
                    else:
                        values = float(val)
                        complete = np.matrix(variables)
                        durations = float(dur)
                    
        
        self.update_jobs(res_file, expts, expts_vals, expts_durs)    
        return res_file , values, complete, durations, pending
    
    def update_jobs(self, res_file, expts, expts_vals, expts_durs):
    
        with open( res_file,'w') as outfile:
            
            for e_ind, e in enumerate(expts):
                output = ' '.join(map(str,np.array(e).squeeze()))
                output = str(expts_vals[e_ind]) + ' ' + str(expts_durs[e_ind]) + ' ' + output + '\n' 
                outfile.write(output)
    
    def prepare_opt_results_file(self, subj):
        """ """
        expt_dir  = os.path.realpath(self.job_dir)
        expt_name = os.path.basename(expt_dir)
        
        res_file = os.path.join(expt_dir, self.job_results_file + '_' + subj)
        with open( self.job_results_file +'_' + subj + '_opt', 'w') as opt_res_file:
            opt_res_file.write("")
            
        with open( res_file, 'r') as infile:
            for line in infile:
                expt = line.split()
                if (len(expt) < 3):
                    continue
        
                expt.pop(0)
                expt.pop(0)
        
                [train_val, temp_dur, learner_params] = self.check_job_complete(expt, subj, self.config)#, gmap)
        
                params_dict = self.generate_params_dict(map(float,expt), subj)#,gmap)
                self.SJR.set_params_dict(params_dict)
                out_file_name = self.SJR.generate_learner_output_file_name(self.SJR.params_list, subj)
                    
                res_file_name = os.path.join(self.SJR.results_opt_path, out_file_name)
                    
                npzfile = np.load(res_file_name + '.npz')
                test_error = npzfile['error']
                test_accuracy = 100 - test_error * 100
                    
                with open( self.job_results_file +'_' + subj + '_opt', 'a') as opt_res_file:
                    opt_res_file.write(' '.join(expt + [str(train_val)] + [str(test_error)] + ['\n']))
                    

    def check_job_complete(self, expt, subj, config):
    
        print expt     
        params_dict = self.generate_params_dict(map(float, expt), subj)
        self.SJR.set_params_dict(params_dict)
        out_file_name = self.SJR.generate_learner_output_file_name(self.SJR.params_list, subj)
        val = None
        dur = None
        learner_params = None
        
        res_file_name = os.path.join(self.SJR.results_path, out_file_name)
        if os.path.isfile(res_file_name):
            current_error, learner_params = self.SJR.my_Learner_Manager.find_cv_error(res_file_name)

            val = current_error
            dur = 100
        
        return val, dur, learner_params
    # And that's it

        
if __name__ == '__main__':

    class Job_Params:
        job_dir = 'BCI_Framework'
        num_all_jobs = 100
        dataset = 'BCICIII3b'
        seed = 1
        classifier_name = 'LogisticRegression'
        feature_extraction = 'BP'
        n_concurrent_jobs = 3
        chooser_module = "GPEIOptChooser"
        grid_size = 1000
#        chooser_module = "RandomChooser"
        
    from time import sleep
    import sklearn

    sp = spearmint_lite(Job_Params)
    complete_jobs = np.zeros(sp.config.configuration['number_of_subjects'])
    
    complete_jobs = sp.main(Job_Params, sp.config, complete_jobs)
    
    while any(complete_jobs < np.ones(sp.config.configuration['number_of_subjects']) * Job_Params.num_all_jobs):
        sleep(300)
        complete_jobs = sp.main(Job_Params, sp.config, complete_jobs)

