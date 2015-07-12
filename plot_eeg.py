import os
import sys
sys.path.append('./BCI_Framework')

import Configuration_BCI

import Main
import Single_Job_runner as SJR
import numpy as np
import Data_Preprocessor as Data_Proc
import matplotlib.pyplot as plt


def calc_power(sig_mat, window_size):
    pass
    sig_mat_pow = np.exp2(sig_mat)
    pow_sig_mat = np.zeros(shape = (sig_mat_pow.shape[0] - window_size, sig_mat_pow.shape[1]))
    for i in range(len(sig_mat_pow) - window_size):
        pow_sig_mat[i,:] = np.sum( sig_mat_pow[i:i+window_size, :], axis = 0)
    
    return pow_sig_mat

if __name__ == '__main__':
    
    channels = 'C34'     
    number_of_CSPs = -1
    params_list = [ 0, 0, 0, 0, -1, 0 ]
    config = Configuration_BCI.Configuration_BCI('./BCI_Framework', 'BCICIV2a', channels)
    for subject_ind, subject in enumerate(config.configuration['subject_names_str']):
        
        dp = Data_Proc.Data_Preprocessor(config, channels, subject, 'raw', number_of_CSPs)
        
        raw_X, raw_Y = dp.load_dataset_train(subject)
        filters = ['delta', 'alpha', 'beta']
        cutoff_frequencies_low =  np.array([[4,8.,16.], [4,8.,16.]])
        cutoff_frequencies_high =  np.array([[8,12.,24.], [8,12.,24.]])
        for filt_number in range(cutoff_frequencies_low.shape[1]):

            
            cutoff_freq_low = cutoff_frequencies_low[:,filt_number]
            cutoff_freq_high = cutoff_frequencies_high[:,filt_number]
            
            raw_X = np.array(raw_X)
            filtered_X = np.copy(raw_X) ##copying the list- always copy lists

            colors = ['r', 'b', 'g', 'c']
            for i in range(raw_X.shape[1]):
                
                filtered_X[:,i] = dp.apply_filter(filtered_X[:,i], cutoff_freq_low[i], cutoff_freq_high[i], config.configuration["sampling_rate"])
#                avg = []
#                plt.subplot(2, 3, i + filt_number*2 + 1)
                if i == 0:

#                    plt.subplot(2, 3, 0)

                    f_pow = plt.figure(1)
                    ax_pow = f_pow.add_subplot(2,3,filt_number + 1)
                    f1 = plt.figure(2)
                    ax = f1.add_subplot(2,3,filt_number + 1)
                    ax.set_title('C3 ' + filters[filt_number])
                    ax_pow.set_title('C3 ' + filters[filt_number])
                    
                else:

#                    plt.subplot(2, 3, 3 )
                    f_pow = plt.figure(1)
                    ax_pow = f_pow.add_subplot(2,3,filt_number + 4)
                    f1 = plt.figure(2)
                    ax = f1.add_subplot(2,3, filt_number + 4)
                    ax.set_title('C4 ' + filters[filt_number])
                    ax_pow.set_title('C4 ' + filters[filt_number])
                    
                    
                for class_label_ind, class_label in enumerate(config.configuration['class_labels_list']):
                    sig_mat = np.reshape(filtered_X[raw_Y == int(class_label),i], (int(config.configuration['movement_trial_size_list'][subject_ind]),-1))
                    
                    avg_sig_mat = np.average(sig_mat, axis = 1)
                    pow_sig_mat = calc_power(np.copy(sig_mat), 100)
                    avg_pow_sig_mat = np.average(pow_sig_mat, axis = 1)

#                    plt.figure(1)                    
#                    plt.plot(range(len(avg_sig_mat)),avg_sig_mat, colors[class_label_ind])
#                    plt.figure(2)
#                    plt.plot(range(len(avg_pow_sig_mat)), avg_pow_sig_mat, colors[class_label_ind])

                    ax_pow.plot(range(len(avg_pow_sig_mat)), avg_pow_sig_mat, colors[class_label_ind])
                    ax.plot(range(len(avg_sig_mat)),avg_sig_mat, colors[class_label_ind])
#                    plt.plot(range(len(avg)), sig_mat[:,2], colors[class_label_ind])
                    print sig_mat.shape, 4**i + filt_number
                    
                
        plt.show()

        print 3
        
