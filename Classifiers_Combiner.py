import os
from sklearn import linear_model
import sys
os.chdir('../bci_framework')
sys.path.append('./BCI_Framework')

import Main
import Configuration_BCI

import Single_Job_runner as SJR
import numpy as np
import spearmint_lite
import Learner_Manager
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

import Learner_Factory
import math
from scipy.stats import mode
from os import listdir
from os.path import isfile, join
import shutil
import numpy as np

class Classifiers_Combiner:

    final_results_folder = '../jne_final_results/'
    aggregation_algorithms = ['MLR', 'VOTE', 'AVERAGE', 'MIN']

    def __init__(self, file_names, feature, classifier, dataset, subject, n_classifiers, optimization_type, bo_selection_type, config):

        # self.true_labels_folder = res_folder

        self.window_size_for_termination = 20
        self.termination_threshold = math.exp(-4)
        self.file_names = file_names
        self.feature = feature
        self.classifier = classifier
        self.dataset = dataset
        self.subject = subject
        self.number_of_classifiers = n_classifiers
        self.optimization_type = optimization_type

        self.bo_selection_type = bo_selection_type

        self.final_accuracies_prediction_dcit = {'MLR':[], 'MIN':[], 'VOTE':[], 'AVERAGE':[]}

        self.config = config

    def aggregate_MLR(self, final_prediction_matrix_on_whole_training, final_prediction_matrix, y_train, y_test):

        final_accuracy_test, final_accuracy_train_cv = 0, 0
        final_accuracy_classifier_index = -1
        cv_accuracies = []
        test_accuracies = []

        x_train = [[]]
        x_test = [[]]
        for class_ind in range(1, self.config.configuration['number_of_classes']):
            x_train.append([])
            x_test.append([])

        for classifier_index in range(self.number_of_classifiers):

            test_predictions = np.zeros(shape = (self.config.configuration['number_of_classes'], len(y_test)))
            train_predictions_cv = np.zeros(shape = (self.config.configuration['number_of_classes'], len(y_train)))
            for class_ind in range(self.config.configuration['number_of_classes']):

                x_train[class_ind].append(final_prediction_matrix_on_whole_training[classifier_index][:,class_ind])
                x_test[class_ind].append(final_prediction_matrix[classifier_index][:,class_ind])

                X = np.array(x_train[class_ind]).T
                X_test = np.array(x_test[class_ind]).T

                cv_accuracy = 0
                cv = StratifiedKFold(y = y_train, n_folds = self.config.configuration["number_of_cv_folds"])
                for train_index, test_index in cv:

                    X_train_cv, X_test_cv = X[train_index], X[test_index]
                    y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

                    clf = linear_model.Lasso(alpha = 0.01, positive=True)

                    # Train the model using the training sets
                    clf.fit(X_train_cv, y_train_cv)
                    train_predictions_cv[class_ind,test_index] = clf.predict(X_test_cv)

                # Train the model using the training sets
                clf = linear_model.Lasso(alpha = 0.01, positive=True)

                if self.config.configuration['number_of_classes'] > 2:
                    new_labels = np.copy(y_train)
                    new_labels[new_labels != self.config.configuration['class_labels_list'][class_ind]] = 0
                    new_labels[new_labels == self.config.configuration['class_labels_list'][class_ind]] = 1
                    clf.fit(X, new_labels)
                else:
                    clf.fit(X, y_train)

                test_predictions[class_ind,:] = clf.predict(X_test)

            test_accuracy_till_now = sum(np.argmax(test_predictions, axis= 0) == y_test -1)/float(len(y_test))
            cv_accuracy_till_now = sum(np.argmax(train_predictions_cv, axis= 0) + 1 == y_train)/float(len(y_train))

            test_accuracies.append(test_accuracy_till_now)
            cv_accuracies.append(cv_accuracy_till_now)

            #termination condition
            if classifier_index > self.window_size_for_termination:
                final_accuracy_test, final_accuracy_train_cv, final_accuracy_classifier_index = self.check_threshold(cv_accuracies, classifier_index,
                                                                                                                     test_accuracy_till_now, final_accuracy_test,
                                                                                                                     final_accuracy_train_cv, final_accuracy_classifier_index)
                # if final_accuracy_classifier_index != -1:
                    # break

        if final_accuracy_classifier_index == -1:
            final_accuracy_test, final_accuracy_train_cv = test_accuracies[self.number_of_classifiers - 1], cv_accuracies[self.number_of_classifiers - 1]
            final_accuracy_classifier_index = self.number_of_classifiers

        print self.bo_selection_type, self.subject, final_accuracy_test, final_accuracy_train_cv, final_accuracy_classifier_index
        with open('../n_classifiers.txt', 'a') as f:
            f.write(self.subject + self.bo_selection_type + str(final_accuracy_classifier_index) + '\n')
        return final_accuracy_test, final_accuracy_train_cv, final_accuracy_classifier_index, cv_accuracies, test_accuracies

    def check_threshold(self, cv_accuracies, classifier_index, test_accuracy_till_now,
                        prev_final_accuracy_test, prev_final_accuracy_train_cv, prev_final_accuracy_classifier_index):

        max_objective = np.max(cv_accuracies[classifier_index - self.window_size_for_termination: classifier_index])
        min_objective = np.min(cv_accuracies[classifier_index - self.window_size_for_termination: classifier_index])
        initial_objective = cv_accuracies[1]

        threshold = (max_objective - min_objective)/float(initial_objective)
        epsilon = self.termination_threshold

        if threshold < epsilon:
            final_accuracy_test, final_accuracy_train_cv = test_accuracy_till_now, cv_accuracies[classifier_index]
            final_accuracy_classifier_index = classifier_index
            return final_accuracy_test, final_accuracy_train_cv, final_accuracy_classifier_index
        else:
            return prev_final_accuracy_test,prev_final_accuracy_train_cv, prev_final_accuracy_classifier_index


    def aggregate_AVG(self, cv_prediction_matrix, final_prediction_matrix, y_train, y_test):

        final_accuracy_test, final_accuracy_train_cv = 0, 0
        final_accuracy_classifier_index = -1

        cv_accuracies = []
        test_accuracies = []
        sum_of_cv_preds = np.zeros(shape=( len(y_train), self.config.configuration['number_of_classes']))
        sum_of_test_preds = np.zeros(shape=( len(y_test), self.config.configuration['number_of_classes']))
        for classifier_index in range(self.number_of_classifiers):

            sum_of_cv_preds = np.add(sum_of_cv_preds, cv_prediction_matrix[classifier_index])
            cv_accuracy_till_now = 0

            cv = StratifiedKFold(y = y_train, n_folds = self.config.configuration["number_of_cv_folds"])
            for train_index, test_index in cv:
                y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

                cv_predictions_till_now = np.argmax(sum_of_cv_preds, axis = 1)[test_index]
                cv_accuracy_till_now += np.sum(cv_predictions_till_now + 1 == y_test_cv)/float(len(y_test_cv))

            cv_accuracy_till_now /= float(self.config.configuration['number_of_cv_folds'])
            cv_accuracies.append(cv_accuracy_till_now)

            # print str(classifier_index) + " cv accuracy: " + str(cv_accuracy_till_now)

            sum_of_test_preds = np.add( sum_of_test_preds, final_prediction_matrix[classifier_index])
            test_predictions_till_now = np.argmax(sum_of_test_preds, axis = 1)
            test_accuracy_till_now = np.sum(test_predictions_till_now + 1 == y_test)/float(len(y_test))

            test_accuracies.append(test_accuracy_till_now)
            # print "test predictions till now: " + str(test_accuracy_till_now)

            #termination condition
            if classifier_index > self.window_size_for_termination:
                final_accuracy_test, final_accuracy_train_cv, final_accuracy_classifier_index = self.check_threshold(cv_accuracies, classifier_index,
                                                                                                                     test_accuracy_till_now, final_accuracy_test,
                                                                                                                     final_accuracy_train_cv, final_accuracy_classifier_index)
                # if final_accuracy_classifier_index != -1:
                #     break

        if final_accuracy_classifier_index == -1:
            final_accuracy_test, final_accuracy_train_cv = test_accuracies[self.number_of_classifiers - 1], cv_accuracies[self.number_of_classifiers - 1]
            final_accuracy_classifier_index = self.number_of_classifiers

        print self.bo_selection_type, self.subject, final_accuracy_test, final_accuracy_train_cv, final_accuracy_classifier_index
        with open('../n_classifiers.txt', 'a') as f:
            f.write(self.subject + self.bo_selection_type + str(final_accuracy_classifier_index) + '\n')
        return final_accuracy_test, final_accuracy_train_cv, final_accuracy_classifier_index, cv_accuracies, test_accuracies


    def aggregate_VOTE(self, cv_prediction_matrix, final_prediction_matrix, y_train, y_test):

        final_accuracy_test, final_accuracy_train_cv = 0, 0
        final_accuracy_classifier_index = -1

        cv_accuracies = [0]
        test_accuracies = []

        #claculate acccuracy for the first classifier
        test_accuracy_till_now = np.sum(final_prediction_matrix[0] + 1 == y_test)/float(len(y_test))
        test_accuracies.append(test_accuracy_till_now)

        for classifier_index in range(1, self.number_of_classifiers):

            cv_accuracy_till_now = 0
            cv = StratifiedKFold(y = y_train, n_folds = self.config.configuration["number_of_cv_folds"])
            cv_labels_till_now = mode(cv_prediction_matrix[0:classifier_index], axis = 0)[0][0]
            for train_index, test_index in cv:
                y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

                cv_accuracy_till_now += np.sum(cv_labels_till_now[test_index] + 1 == y_test_cv)/float(len(y_test_cv))

            cv_accuracy_till_now /= float(self.config.configuration['number_of_cv_folds'])
            cv_accuracies.append(cv_accuracy_till_now)

            # print str(classifier_index) + " cv accuracy: " + str(cv_accuracy_till_now)
            test_labels_till_now = mode(final_prediction_matrix[0:classifier_index], axis = 0)[0][0]
            test_accuracy_till_now = np.sum(test_labels_till_now + 1 == y_test)/float(len(y_test))

            test_accuracies.append(test_accuracy_till_now)
            # print "test predictions till now: " + str(test_accuracy_till_now)

            #termination condition
            if classifier_index > self.window_size_for_termination:
                final_accuracy_test, final_accuracy_train_cv, final_accuracy_classifier_index = self.check_threshold(cv_accuracies, classifier_index,
                                                                                                                     test_accuracy_till_now, final_accuracy_test,
                                                                                                                     final_accuracy_train_cv, final_accuracy_classifier_index)
                # if final_accuracy_classifier_index != -1:
                #     break

        if final_accuracy_classifier_index == -1:
            final_accuracy_test, final_accuracy_train_cv = test_accuracies[self.number_of_classifiers - 1], cv_accuracies[self.number_of_classifiers - 1]
            final_accuracy_classifier_index = self.number_of_classifiers

        print self.bo_selection_type, self.subject, final_accuracy_test, final_accuracy_train_cv, final_accuracy_classifier_index
        with open('../n_classifiers.txt', 'a') as f:
            f.write(self.subject + self.bo_selection_type + str(final_accuracy_classifier_index) + '\n')
        return final_accuracy_test, final_accuracy_train_cv, final_accuracy_classifier_index, cv_accuracies, test_accuracies

    def calculate_cv_test_error_for_one_candidates(self, candidate_file_name, classifier_index, cv_prediction_matrix, cv_prediction_matrix_labels,
                                                   final_prediction_matrix, final_prediction_matrix_labels, minimum_classifier_cv_error_index, minimum_classifier_cv_error,
                                                   x_train, y_train, x_test, y_test, classifier_cv_error, learners_params, final_prediction_matrix_on_whole_training):

        y_train_cv_all_predictions = np.zeros(shape=(len(y_train), self.config.configuration['number_of_classes']))
        # y_test_all_predictions = np.zeros(shape=(len(y_test), config.configuration['number_of_classes']))

        #cross validation
        cv = StratifiedKFold(y = y_train, n_folds = self.config.configuration["number_of_cv_folds"])
        for train_index, test_index in cv:

            x_train_cv, x_test_cv = x_train[train_index], x_train[test_index]
            y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]
            #scale data
            x_train_cv, x_test_cv = self.scale_data(x_train_cv, x_test_cv)
            #create classifier
            y_pred_train_cv, y_pred_test_cv = self.train_classifier(learners_params, x_train_cv, y_train_cv, x_test_cv)
            y_train_cv_all_predictions[test_index, :] = y_pred_test_cv

        cv_prediction_matrix.append(y_train_cv_all_predictions)
        cv_prediction_matrix_labels.append(np.argmax(y_train_cv_all_predictions, axis = 1))

        #train on whole training data and test on whole test data
        x_train, x_test = self.scale_data(x_train, x_test)
        y_pred_train, y_pred_test = self.train_classifier(learners_params, x_train, y_train, x_test)
        final_prediction_matrix.append(y_pred_test)
        final_prediction_matrix_labels.append(np.argmax(y_pred_test, axis = 1))
        final_prediction_matrix_on_whole_training.append(y_pred_train)

        #find classifier with minimum CV error in optimization phase
        if minimum_classifier_cv_error > classifier_cv_error:
            minimum_classifier_cv_error = classifier_cv_error
            minimum_classifier_cv_error_index = classifier_index

        return minimum_classifier_cv_error_index, minimum_classifier_cv_error

    def apply_all_aggregation_algorithms(self):

        for candidate_file_name in self.file_names:
            print candidate_file_name
            cv_prediction_matrix = []
            cv_prediction_matrix_labels = []
            final_prediction_matrix_on_whole_training = []
            final_prediction_matrix = []
            final_prediction_matrix_labels = []
            minimum_classifier_cv_error = 1
            minimum_classifier_cv_error_index = 0

            for classifier_index in range(self.number_of_classifiers):

                with open(candidate_file_name) as candid_file:
                    all_lines = candid_file.readlines()
                    params = all_lines[classifier_index]

                #generate parameters
                classifier_cv_error, params_dict, params_list, learners_params = self.generate_params_dict(params)

                #load training and test datasets
                x_train, y_train, x_test, y_test = self.load_data(params_list)

                minimum_classifier_cv_error_index, minimum_classifier_cv_error = \
                    self.calculate_cv_test_error_for_one_candidates(candidate_file_name, classifier_index, cv_prediction_matrix, cv_prediction_matrix_labels,
                                                                    final_prediction_matrix, final_prediction_matrix_labels, minimum_classifier_cv_error_index, minimum_classifier_cv_error,
                                                                    x_train, y_train, x_test, y_test, classifier_cv_error, learners_params, final_prediction_matrix_on_whole_training)

            AVG_test_error, AVG_train_error, AVG_n_classiifers_used, AVG_cv_accuracies, AVG_test_accuracies  = self.aggregate_AVG(cv_prediction_matrix, final_prediction_matrix, y_train, y_test)
            self.final_accuracies_prediction_dcit['AVERAGE'].append(AVG_test_error)
            VOTE_test_error, VOTE_train_error, VOTE_n_classiifers_used, VOTE_cv_accuracies, VOTE_test_accuracies = self.aggregate_VOTE(cv_prediction_matrix_labels,
                                                                                             final_prediction_matrix_labels, y_train, y_test)
            self.final_accuracies_prediction_dcit['VOTE'].append(VOTE_test_error)

            minimum_classifier_test_error = np.sum(final_prediction_matrix_labels[minimum_classifier_cv_error_index]+1 == y_test)/float(len(y_test))
            self.final_accuracies_prediction_dcit['MIN'].append(minimum_classifier_test_error)

            #MLR
            MLR_test_error, MLR_train_error, MLR_n_classiifers_used, MLR_cv_accuracies, MLR_test_accuracies = self.aggregate_MLR(final_prediction_matrix_on_whole_training, final_prediction_matrix, y_train, y_test)
            self.final_accuracies_prediction_dcit['MLR'].append(MLR_test_error)


        final_results = {}
        for aggragation_method in self.final_accuracies_prediction_dcit.keys():
            mean_val = np.mean(self.final_accuracies_prediction_dcit[aggragation_method])
            std_val = np.std(self.final_accuracies_prediction_dcit[aggragation_method])

            final_results[(aggragation_method, 'MEAN')] = mean_val
            final_results[(aggragation_method, 'STD')] = std_val

        #Plot accuracies as we increase the number of classifiers
        self.plot_accuracies(MLR_cv_accuracies, MLR_test_accuracies, AVG_cv_accuracies, AVG_test_accuracies, VOTE_cv_accuracies, VOTE_test_accuracies)

        return final_results

    def plot_accuracies(self, MLR_cv_accuracies, MLR_test_accuracies, AVG_cv_accuracies, AVG_test_accuracies, VOTE_cv_accuracies, VOTE_test_accuracies):

        x = np.arange(0,80)
        import matplotlib
        matplotlib.use('TkAgg')
        from matplotlib import pyplot as plt
        plt.xlabel('Number of iterations')
        plt.ylabel('Accuracy (%)')
        with plt.style.context('fivethirtyeight'):
            MLRCV,  = plt.plot(x, np.array(MLR_cv_accuracies)*100, color = 'k')
            MLRtest,  = plt.plot(x, np.array(MLR_test_accuracies)*100)
            AVGCV,  = plt.plot(x, np.array(AVG_cv_accuracies)*100)
            AVGtest,  = plt.plot(x, np.array(AVG_test_accuracies)*100)
            VOTECV,  = plt.plot(x, np.array(VOTE_cv_accuracies)*100)
            VOTEtest,  = plt.plot(x, np.array(VOTE_test_accuracies)*100)
            plt.legend( (MLRCV, MLRtest, AVGCV, AVGtest, VOTECV, VOTEtest), ('BO-GP (MLR_CV)', 'BO-GP (MLR_Test)', 'BO-GP (AVG_CV)', 'BO-GP (AVG_Test)', 'BO-GP (VOTE_CV)', 'BO-GP (VOTE_Test)'), loc='lower right', shadow=True)

        plt.show()
        print 3

    def scale_data(self, x_train, x_test):

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

        return x_train, x_test

    def train_classifier(self, learners_params, x_train, y_train, x_test):

        my_learner_factory = Learner_Factory.Learner_Factory(self.config)
        learner = my_learner_factory.create_learner(self.classifier)
        learner.set_params_dict(learners_params)
        learner.fit(x_train, y_train)
        y_pred_train = learner.learner.predict_proba(x_train)
        y_pred_test = learner.learner.predict_proba(x_test)

        return y_pred_train, y_pred_test

    def generate_params_dict(self, params):

        sp = spearmint_lite.spearmint_lite(None, None, self.config, self.optimization_type)

        params_dict = sp.generate_params_dict(map(float, params.split(' ')[2:]), self.subject)
        params_list = SJR.Simple_Job_Runner.create_params_list(params_dict)

        out_file_name = SJR.Simple_Job_Runner.generate_learner_output_file_name( params_list, self.subject)
        results_path = self.config.configuration['results_path_str']
        results_opt_path = self.config.configuration['results_opt_path_str']

        results_path, results_opt_path = SJR.Simple_Job_Runner.set_results_path( results_path, results_opt_path, self.classifier, self.feature,
                                                                         optimization_type = self.optimization_type, BO_selection_type = '')
        res_file_name = os.path.join(results_path, out_file_name)
        my_Learner_Manager = Learner_Manager.Learner_Manager(self.config, self.classifier, self.feature)

        current_error, learner_params = my_Learner_Manager.find_cv_error(res_file_name)

        # params_dict = sp.generate_params_dict(map(float, params), self.subject)
        params_dict = dict(params_dict.items() + learner_params.items())

        return float(params.split(' ')[0]), params_dict, params_list, learner_params

    def load_data(self, params_list):

        # I am assuming the feature files are already there so I do not need to extract features again!
        out_name = SJR.Simple_Job_Runner.generate_learner_output_file_name(params_list, self.subject)
        out_path = os.path.join(self.config.configuration['feature_matrix_dir_name_str'], self.feature, out_name + '.npz')

        A = np.load(out_path)
        x_train, y_train, x_test, y_test =  A['arr_0'][0], A['arr_1'], A['arr_2'][0], A['arr_3']

        return x_train, y_train, x_test, y_test

    @staticmethod
    def write_results_to_file(final_results, datasets, BO_selection_types, feature):

        n_aggregation_methods = len(Classifiers_Combiner.aggregation_algorithms)
        n_bo_types = len(BO_selection_types)

        for dataset in datasets:

            config = Configuration_BCI.Configuration_BCI('BCI_Framework', dataset)
            mean_matrix = np.zeros(shape=(config.configuration['number_of_subjects'], n_bo_types*n_aggregation_methods))
            mean_matrix_file_name = os.path.join(Classifiers_Combiner.final_results_folder, dataset + '_' + feature + classifier + '_MEAN.csv')
            latex_matrix = np.chararray(shape=(config.configuration['number_of_subjects'], n_bo_types*n_aggregation_methods), itemsize = 50) #this is mean +- std
            latex_matrix_file_name = os.path.join(Classifiers_Combiner.final_results_folder, dataset + '_' + feature + classifier + '_LATEX.csv')

            for subject_ind, subject in enumerate(config.configuration['subject_names_str']):

                best_index = [0]
                best_value = 0
                for bo_selection_index, bo_selection in enumerate(BO_selection_types):
                    for aggregation_method_ind, aggregation_method in enumerate(Classifiers_Combiner.aggregation_algorithms):

                        mean_value = final_results[dataset][(subject, bo_selection)][(aggregation_method, 'MEAN')] * 100
                        std_value = final_results[dataset][(subject, bo_selection)][(aggregation_method, 'STD')] * 100

                        mean_matrix[subject_ind, bo_selection_index*n_aggregation_methods + aggregation_method_ind] = float("{0:.2f}".format(mean_value))

                        latex_matrix[subject_ind, bo_selection_index*n_aggregation_methods + aggregation_method_ind] = str("{0:.2f}".format(mean_value)) + '$\pm$' + str("{0:.2f}".format(std_value))

                        if mean_value > best_value:
                            best_value = mean_value
                            best_index = [bo_selection_index*n_aggregation_methods + aggregation_method_ind]
                        elif mean_value == best_value:
                            best_index.append(bo_selection_index*n_aggregation_methods + aggregation_method_ind)

                for ind in best_index:
                    latex_matrix[subject_ind, ind] = "\\cellcolor{blue!25}" + latex_matrix[subject_ind, ind]


            np.savetxt(mean_matrix_file_name, mean_matrix, delimiter=',')
            np.savetxt(latex_matrix_file_name, latex_matrix, delimiter=',', fmt="%s")


if __name__ == '__main__':

    candidates_folder = '../Candidates'
    number_of_classifiers = 80
    number_of_runs = 1
    classifier_name = 'LogisticRegression'#'LDA'#'LogisticRegression'
    datasets = ['BCICIV2a']#['BCICIV2a'] #['BCICIII3b']#, 'BCICIV2b', 'BCICIV2a' ]
    classifier = 'LogisticRegression'#'LogisticRegression'
    feature = 'BP'# 'morlet'] #morlet for type 2 and 4 does not work!!!!!
    optimization_types_dict = {('BCICIII3b','BP'):[2], ('BCICIII3b','morlet'):[1], ('BCICIV2b','BP') : [2], ('BCICIV2b','morlet') : [1],
                               ('BCICIV2a','BP') : [4], ('BCICIV2a','morlet') : [3]}
    BO_selection_types = [ "GPEIOptChooser", "RandomForestEIChooser", "RandomChooser"]


    #first copy files in "GPEIOptChooser", "RandomForestEIChooser", "RandomChooser" folders to their parent folder
    # for dataset in datasets:
    #     for bo_selection_method in BO_selection_types:
    #         src_folder = os.path.join('..', dataset, classifier, feature, optimization_types_dict(dataset, feature), bo_selection_method)
    #         dest_folder = os.path.join('..', dataset, classifier, feature, optimization_types_dict(dataset, feature))
    #         onlyfiles = [ f for f in listdir(src_folder) if isfile(join(src_folder,f)) ]
    #         for my_file in onlyfiles:
    #             shutil.copyfile(os.path.join(src_folder, my_file), os.path.join(dest_folder, my_file))

    final_results = {}
    for dataset in datasets:
        my_config =  Configuration_BCI.Configuration_BCI('BCI_Framework', dataset)
        final_results_for_this_dataset = {}
        for subject in my_config.configuration['subject_names_str']:
            for bo_type in BO_selection_types:
                file_names = []
                for run_number in range(1,number_of_runs+1):
                    #generate list of candidate candidate file names
                    file_name = 'results_{0}_{1}{2}_' + classifier + '_{3}.dat_{4}'
                    file_name = file_name.format(optimization_types_dict[(dataset, feature)][0], bo_type,run_number, feature, subject)
                    file_names.append( os.path.join(candidates_folder, file_name))

                classifier_combiner = Classifiers_Combiner(file_names, feature, classifier,dataset, subject,
                                                           number_of_classifiers, optimization_types_dict[(dataset, feature)][0], bo_type, my_config)

                final_results_for_this_subj_this_bo_type = classifier_combiner.apply_all_aggregation_algorithms()
                final_results_for_this_dataset[(subject, bo_type)] = final_results_for_this_subj_this_bo_type

        print final_results_for_this_dataset

        final_results[dataset] = final_results_for_this_dataset

    Classifiers_Combiner.write_results_to_file(final_results, datasets, BO_selection_types, feature)

