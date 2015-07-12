#!/usr/bin/env python
# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt

def bar_plotter(labels, y_titles, accuracies, sub_names, type = 'COMPARE_CLASSIFIERS'):
    
    plt.switch_backend('TkAgg')  

    if type == 'COMPARE_CLASSIFIERS':
        n_beans = accuracies.shape[0]
        n_plots = accuracies.shape[1]
    elif type == 'COMPARE_FEATURES':
        n_beans = accuracies.shape[1]
        n_plots = accuracies.shape[0]
        
    default_colors = ['r', 'y', 'g', 'b', 'c','m', 'k', 'b', 'r']
    default_widths = { 3:.35 , 8:0.15, 9:0.1, 4:0.28}
    n_beans = len(labels)
    n_subjects = len(sub_names)
    
    rects_list_all = []
    rects_list_each_plot = []
    
    for plot_ind in range(n_plots):

        rects_list_each_plot = []

        fig, ax = plt.subplots()
        ind = np.arange(1,n_beans*2+1, 2)
        for subj_ind in range(n_subjects):
#             for plot_ind in range(n_plots):
            if type == 'COMPARE_CLASSIFIERS':
                rects = ax.bar(ind + subj_ind*default_widths[n_subjects], accuracies[:, plot_ind, subj_ind], default_widths[n_subjects], color=default_colors[subj_ind])
            else:
                rects = ax.bar(ind + subj_ind*default_widths[n_subjects], accuracies[plot_ind, :, subj_ind], default_widths[n_subjects], color=default_colors[subj_ind])
            rects_list_each_plot.append(rects)
#             rects_list_all.append(rects_list_each_plot)
        # add some
#     for plot_ind in range(n_plots):
        
        rects_list = rects_list_each_plot
        ax.set_ylabel('Accuracies')
        ax.set_title(y_titles[plot_ind])
        ax.set_xticks(ind+default_widths[n_subjects] * n_subjects/2.0)
        ax.set_xticklabels( labels )
        
#     ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )
        ax.legend( [rects[0] for rects in rects_list], sub_names )
    
        
        for rects in rects_list:
            autolabel(rects, ax) 
    
    plt.show()

def autolabel(rects, ax):
        # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
            ha='center', va='bottom')
        

if __name__ == '__main__':


    N = 5
    menMeans = (20, 35, 30, 35, 27)
    menStd =   (2, 3, 4, 1, 2)
    
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35       # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, menMeans, width, color='r', yerr=menStd)
    
    womenMeans = (25, 32, 34, 20, 25)
    womenStd =   (3, 5, 2, 3, 3)
    rects2 = ax.bar(ind+width, womenMeans, width, color='y', yerr=womenStd)
    
    # add some
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('G1', 'G2', 'G3', 'G4', 'G5') )
    
    ax.legend( (rects1[0], rects2[0]), ('Men', 'Women') )
    
    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                    ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    plt.show()