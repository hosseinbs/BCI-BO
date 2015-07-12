import numpy as np
import matplotlib.pyplot as plt

# make a little extra space between the subplots
plt.subplots_adjust(wspace=0.5)

dt = 0.01
t = np.arange(0, 40, dt)
nse1 = 5 * np.random.randn(len(t))              # white noise 1
nse2 = np.random.randn(len(t))                 # white noise 2
r = np.exp(-t/0.05)

cnse1 = np.convolve(nse1, r, mode='same')*dt   # colored noise 1

# two signals with a coherent part and a random part
s1 = 0.01*np.sin(2*np.pi*10*t) + cnse1
#s = 0.01*np.sin(2*np.pi*10*t) + cnse1

for i in range(1,31):
    nse1 = (i/2.0) * np.random.randn(len(t))              # white noise 1

    cnse1 = np.convolve(nse1, r, mode='same')*dt 
    s1 = 0.01*np.sin(2*np.pi*10*t) + cnse1 + i
#    s = s + s1
    plt.plot(t, s1, 'b-')#, t, s2, 'g-')
    
#plt.title('The decomposed signal for channel i', fontsize=28)
   
#plt.plot(t, s/500., 'b-')#, t, s2, 'g-')  
plt.xlim(0,16)
plt.locator_params(axis='x', tight=True,  nbins=4)
plt.xticks( range(17), ('', '', '', '','1', '', '', '','2', '', '', '','3', '', '', '','4', '', '', '',) , fontsize=30) 
plt.ylim(0,31)
plt.xlabel('Time Interval', fontsize=30)
plt.ylabel('s1 and s2')
plt.grid(True)

rect = plt.Rectangle((2.5, 2.8), 3.2, 4.3, edgecolor="red" ,facecolor="#efefaa")
plt.gca().add_patch(rect)

plt.text(4.1, 1.0, "$T$", size=30,   ha="center", va="center" , fontsize=30)
plt.text(2.0, 5.1, "$F$", size=30,  rotation=90, ha="center", va="center" , fontsize=30)

rect = plt.Rectangle((6.5, 4.4), 3.2, 4.3, edgecolor="red" ,facecolor="#efefaa")
plt.gca().add_patch(rect)

rect = plt.Rectangle((12.5, 22.4), 3.2, 4.3, edgecolor="red" ,facecolor="#efefaa")
plt.gca().add_patch(rect)
#s2 = 0.01*np.sin(2*np.pi*10*t) + cnse2

#plt.subplot(211)

#plt.subplot(212)
#cxy, f = plt.cohere(s1, s2, 256, 1./dt)
plt.ylabel('Frequency', fontsize=28)
plt.yticks( fontsize=30)
#plt.show()


import pylab as px


def half_brace(x, beta):
    x0, x1 = x[0], x[-1]
    y =  -1 *  (1/(1.+np.exp(-1*beta*(x-x0))) + 1/(1.+np.exp(-1*beta*(x-x1))))
    return y

xmax, xstep = 3.1, .01
xaxis = np.arange(0, xmax/2, xstep)
y0 = half_brace(xaxis, 26.00)
y = np.concatenate((y0, y0[::-1]))

px.plot(np.arange(0, xmax, xstep) + 2.55, y+3.2, color='black')


xmax, xstep = 4., .001
xaxis = np.arange(0, xmax/2, xstep)
y0 = half_brace(xaxis, 26.00)
y = np.concatenate((y0, y0[::-1]))

px.plot( y/3.0+2.6, np.arange(0, xmax, xstep) + 3, color='black')




px.show()