import numpy as np
import matplotlib.pyplot as plt
import math
#from pylab  import figure
import pylab as P
    
def plotter(y_vector, min_acc, max_acc, fig_ind, labels):
        
#    figure(fig_ind)
    f1 = plt.figure(fig_ind)
    ax = f1.add_subplot(111)
#    fig, ax = plt.subplots()

    # plot control points and connecting lines
    x = (1,2,3,4,5,6,7)
    
    markers = ['o', 'v','^','s','p','*','h','H','+','x','D','d']
    colors = ['b','g','r','c','m','y', 'k', 'b','g','r','c','m','y', 'k',]
    for ind, subj_y in enumerate(y_vector):
        
        line, = ax.plot(x, subj_y, colors[ind]+markers[ind]+'-', markersize= 14, label = labels[ind], linewidth=5.0)
    
    
#    legend( (p1, p2), ('12:30', '23:30') )
    plt.legend(fontsize = 27, loc=4)
    
#    plt.legend(bbox_to_anchor=(0., 1., 1., .1), loc=3,
#       ncol=3, mode="expand", borderaxespad=0., fontsize = 20)
    
    ax.yaxis.grid(True, 'major')

    
    y_ticks = []
    for i in np.arange(min_acc,max_acc,1):
        if math.floor(i) == i and i%4 == 0:
            y_ticks.append(str(int(i))) 
        else:
            y_ticks.append('')
     
    plt.xticks( range(9), ('','BST', 'LR', 'RF','SVM', 'LDA', 'QDA','MLP', '') , fontsize=25)
    plt.yticks( np.arange(min_acc,max_acc,1), y_ticks, fontsize = 21)
    
    plt.ylabel('Accuracy (%)', fontsize=25)

    
def BO_plotter(y_vector, min_acc, max_acc, fig_ind, labels):
        
#    figure(fig_ind)
    f1 = plt.figure(fig_ind)
    ax = f1.add_subplot(111)
#    fig, ax = plt.subplots()

    # plot control points and connecting lines
    x = (1,2,3)
#    x = (1,2,3,4)
    
    markers = ['o','v','^','s','p','*','H','+','x','h','D','d']
#    colors = ['b','g','r','b','g','r','b','g','r','c','m','y', 'k', 'b','g','r','c','m','y', 'k',]
    colors = ['b','b','b','b','g','r','b','g','r','c','m','y', 'k', 'b','g','r','c','m','y', 'k',]

    for ind, subj_y in enumerate(y_vector):
        
        line, = ax.plot(x, subj_y, colors[ind]+markers[ind]+'-', markersize= 14, label = labels[ind], linewidth=4.0, linestyle="--")
    
    plt.legend(fontsize = 18, loc=3)
#    plt.legend(bbox_to_anchor=(0., 0.92, 1., .1), loc=3,
#       ncol=2, mode="expand", borderaxespad=0.)
    ax.yaxis.grid(True, 'major')

    
    y_ticks = []
    for i in np.arange(min_acc,max_acc,0.5):
        if math.floor(i) == i and i%2 == 0:
            y_ticks.append(str(int(i))) 
        else:
            y_ticks.append('')
     
#    plt.xticks( range(6), ('','BP', 'LogBP', 'RAW', 'wackerman', '') , fontsize=25)
    plt.xticks( range(5), ('','BP', 'LogBP', 'wackerman', '') , fontsize=25)
    plt.yticks( np.arange(min_acc,max_acc,0.5), y_ticks, fontsize = 20)
    
    plt.ylabel('Accuracy', fontsize=25)


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

def bar_chart_plotter(bar_chart_mat_opt, classifiers_list, features_list):
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_size, y_size = bar_chart_mat_opt.shape
    
    x, y = np.random.rand(2, 100) * 4
    hist, xedges, yedges = np.histogram2d(x, y, bins=4)
    
    elements = (len(xedges) - 1) * (len(yedges) - 1)
    
#    xpos, ypos = np.meshgrid(xedges[:-1]+0.25, yedges[:-1]+0.25)
    
    xpos, ypos = np.meshgrid(range(x_size), range(y_size))
    
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(x_size*y_size)
    dx = 0.5 * np.ones_like(zpos)
    dy = dx.copy()
    dz = bar_chart_mat_opt.flatten()
    
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='b', zsort='average')
    
#    plt.show()    
    print 3
    
    
#    import numpy as np
#    import matplotlib.pyplot as plt
    
    N = 6
    
    ind = np.arange(N)  # the x locations for the groups
    width = 0.2       # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, bar_chart_mat_opt[:,0], width, color='r')
    
    rects2 = ax.bar(ind+width, bar_chart_mat_opt[:,1], width, color='g')

    rects3 = ax.bar(ind+2*width, bar_chart_mat_opt[:,2], width, color='b')

    # add some
    ax.set_ylabel('Percentage' , fontsize = 25)
    ax.set_title('Percentage of success of each classifier for each feature')
    ax.set_xticks(ind+width*1.5 )
    
    ax.set_xticklabels( tuple(classifiers_list), fontsize = 20 )
    
    ax.legend( (rects1[0], rects2[0], rects3[0]), tuple(features_list) , fontsize = 16, loc=2)
    
    def autolabel(rects):
    # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.01*height, '%d'%int(height),
                ha='center', va='bottom', fontsize = 18)
    
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    plt.show()
    
    print 4