"""
Pyplot animation example.

The method shown here is only for very simple, low-performance
use.  For more demanding applications, look at the animation
module and the examples that use it.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
os.chdir('../bci_framework')
sys.path.append('./BCI_Framework')

import Main
import Configuration_BCI

import Single_Job_runner as SJR
import numpy as np
import itertools


dataset = 'BCICIII3b'
config = Configuration_BCI.Configuration_BCI('BCI_Framework', dataset)

for subj_ind, subject in enumerate(config.configuration['subject_names_str']):

    x = np.arange(config.configuration['movement_trial_size_list'][subj_ind])
    
    y = np.arange(620)
    
    z = np.zeros((len(y),len(x)))    
    
    sum_count = np.ones((len(y),len(x)))
    
    file_name = 'results_' + dataset + '.dat_' + subject
    with open(os.path.join('BCI_Framework',file_name), 'r') as candidates_file:
        
        all_candidates = candidates_file.readlines()
#     all_cands = np.loadtxt(os.path.join('BCI_Framework',file_name))
    for candidate in all_candidates:
        print candidate
        candidate = candidate.strip()
        if 'P' not in candidate:
            
            candidates_array = map(float, candidate.split())
            
            weight = candidates_array[0]
            
            low_time = candidates_array[2]
            high_time = candidates_array[3]
            low_freq = 20*candidates_array[4]
            high_freq = 20*candidates_array[5]
    
            z[-high_freq:-low_freq, low_time:-high_time] += weight
#             sum_count[-high_freq:-low_freq, low_time:-high_time] += 1 

    z = z/sum_count
#     z = z * 100
    np.savetxt('z.csv', z, fmt='%.2f', delimiter = ',')
    p = plt.imshow(z)
    fig = plt.gcf()
    plt.clim()   # clamp the color limits
    plt.title("Time/Frequency importance of the the signal")
    
    mv_length = config.configuration['movement_trial_size_list'][subj_ind]/float(config.configuration['sampling_rate'])
    tick_locs = np.arange(mv_length+1) * float(config.configuration['sampling_rate'])
    tick_lbls = np.arange(mv_length+1)
    plt.xticks(tick_locs, tick_lbls)

    
    tick_locs = np.arange(32) * 20
    tick_lbls = list(reversed(range(32)))
    plt.yticks(tick_locs, tick_lbls)

    plt.colorbar()
    
    plt.show()


# p = plt.imshow(z)
# fig = plt.gcf()
# plt.clim()   # clamp the color limits
# plt.title("Boring slide show")
# 
# plt.show()

# for i in xrange(5):
#     if i==0:
#         p = plt.imshow(z)
#         fig = plt.gcf()
#         plt.clim()   # clamp the color limits
#         plt.title("Boring slide show")
#     else:
#         z = z + 2
#         p.set_data(z)
# 
#     print("step", i)
#     plt.pause(0.5)