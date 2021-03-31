#!/usr/bin/env python
# coding: utf-8

# # Monte Carlo Methods
# 
# This exercise address the use of “random” numbers in Monte Carlo techniques. These are often the fastest or most straightforward way of tackling complicated problems in computational analysis.
# 
# You should use the random number routines included in numpy.random :
# https://docs.scipy.org/doc/numpy-1.14.0/reference/routines.random.html
# 
# This library uses the "Mersenne Twister" algorithm internally, which is a modern, well behaved, pseudo-random number generator. Note that, by default, the generator will be initialised with a "seed" based on the time when the programme is started - giving a different sequence of numbers every time you run the programme. You may find it useful, while debuggging, to manually set a fixed "seed" at the start of the programme.  This will result in an identical sequence of random numbers, every time you run the programme.
# 
# ## Q1 - Generating Distributions
# 
# In practise we usually want to generate floating point numbers with a particular distribution. Numpy.random includes several built-in distributions, however we often need to write our own. Two methods for achieving this were discussed in Lecture 3 :
# 1) an analytical function derived from the cumulative distribution function of the desired distribution.
# 2) the accept/reject method
# 
# ### 1a)
# Write code to generate random angles $\theta$, between 0 and $\pi$, with a probability distribution proportional to ${\rm sin}(\theta)$. You should write one routine based on the analytical method 1), and another using the accept/reject method. Both routines should use _numpy.random.random()_ to generate floating point numbers with a distribution between 0 and 1, and convert this to the desired ${\rm sin}(\theta)$ distribution.

# In[75]:


import numpy as np
import math as m
import matplotlib.pyplot as plt
import collections as c
import time 

def sine(n):
    x_vals=[i*m.pi/n for i in range(n)]
    y_vals=[.01955*n*m.sin(i*m.pi/n) for i in range(n)]         
    return x_vals, y_vals

def Analytic(n,r):
    theta_vals=[]
    Time=[]
    t=0
    for i in range(n):
        start=time.time()
        C=np.random.random(1)
        theta=m.acos(1-2*C)
        theta_vals.append(round(theta,r))
        end=time.time()
        t=t+(end-start)
        Time.append(t)
    number=[i for i in range(n)]
    print("Total Time of Analytical Method:",t)
    return theta_vals, number, Time

def accept_reject(n,r):
    theta_vals=[]
    Time=[]
    t=0
    for i in range(n):
        start=time.time()
        x=0
        y=1
        while y>x:
            theta=m.floor(m.pi*np.random.random(1)*10**r)/10**r
            x=m.sin(theta)
            y=np.random.random(1)
        theta_vals.append(theta)
        end=time.time()
        t=t+end-start
        Time.append(t)
    number=[i for i in range(n)]
    print("Total Time of Accept/Reject Method:",t)
    return theta_vals, number, Time

def Accuracy(n,r):
    N=3100
    c_anal, freq_anal=np.unique(Analytic(n,r)[0], return_counts=True)
    c_acc, freq_acc=np.unique(accept_reject(n,r)[0], return_counts=True)
    freq_sine=sine(N)[1]
    accuracy_anal=[freq_anal[i]-freq_sine[i] for i in range(N)]
    accuracy_acc=[freq_acc[i]-freq_sine[i] for i in range(N)]
    sum_anal=np.sum(accuracy_anal)
    sum_acc=np.sum(accuracy_acc)
    std_anal=(sum_anal**2/N)**.5
    std_acc=(sum_acc**2/N)**.5
    return accuracy_anal, accuracy_acc, std_anal, std_acc


# ### 1b)
# Now verify that the two routines produce the desired distribution, and evaluate their performance (in both cpu terms and accuracy).  Discuss your results in the text cell below.

# In[78]:


n=120000
r=3

S=sine(n)

A=Analytic(n,r)
plt.hist(A[0], 80, histtype='step', label='Analytical')
plt.plot(S[0], S[1], c='r', linewidth=0.5, label="Sine curve")
plt.xlabel("Value of Theta (Radians)", fontsize=13)
plt.ylabel("Frequency of Value", fontsize=13)
plt.title("The distribution as iterated by the Analytical method")
plt.legend()
plt.show()
print("The blue dots in the graph above is the distribution due to the analytical method.")
print("The red line is a sine curve with domain between 0 and pi and an amplitude scaled to model the distribution.")
print("------------------------------------------------------------------------------------------------------------")
print("")

B=accept_reject(n,r)
plt.hist(B[0], bins=80, histtype='step', label="Accept/Reject")
plt.plot(S[0], S[1], c='r', linewidth=0.5, label="Sine curve")
plt.xlabel("Value of Theta (Radians)", fontsize=13)
plt.ylabel("Frequency of Value", fontsize=13)
plt.title("The distribution as iterated by the Accept Reject method")
plt.legend()
plt.show()
print("The blue dots in the graph above is the distribution due to the accept/reject method.")
print("The red line is a sine curve with domain between 0 and pi and an amplitude scaled to model the distribution.")
print("------------------------------------------------------------------------------------------------------------")
print("")

plt.plot(A[1], A[2], label="Analytical", linestyle="dashed", color="b", linewidth=0.5)
plt.plot(B[1], B[2], label="Accept/Reject", color="r")
plt.ylabel("Time (s)", fontsize=13)
plt.xlabel("Number of iterations", fontsize=13)
plt.title("CPU time of each process over the number of iterations.")
plt.legend()
plt.show()
print("This plot shows the time taken for both methods. As you can see the Accept/Reject method takes much longer")
print("then the Analytical method, this was expected since the Accept/Reject method will do more total iterations")
print("due to the fact that it will reject many of them.")
print("------------------------------------------------------------------------------------------------------------")
print("")
a=Accuracy(n,r)
plt.figure()
plt.plot(sine(3100)[0],a[0], label='Analytical', color='b', linewidth=0.5)                                                        
plt.plot(sine(3100)[0],a[1], label='Accept/Reject', color='r', linewidth=0.5)
plt.ylabel('Error in the Methods', fontsize=12)
plt.xlabel('Angle (rad)', fontsize=14)
plt.title('Inaccuracy of Methods', fontsize=14)
plt.legend()
plt.show()
print("Standard Deviation of the Analytical Method: ", a[2])
print("Standard Deviation of the Accept/Reject Method: ", a[3])
print("These two methods have similar accuracies.")


# # Results Discussion:
# 
# 
# The two methods both give the correct probability distribution of sin(theta). It can be seen that the Accept/Reject method takes considerably longer then the Analytical method. As stated above this is expected since the Accept/Reject method will take more iterations due to the fact that it rejects many of them and still needs to accept the same number as the Analytical method.
# My hypothesis was that the ratio of gradients would be equal 1:(2/pi) for time of Accept/Reject against time of Analytical respectively. This was because the Accept/Reject method is expected to do more pi/2 more iterations then the Analytical method. However it does not turn out to be this ratio, and is in fact closer to 1:0.45.
# 
# In terms of the accuracy of the two methods, I plotted the difference of the two methods at a range of angles. I then also took the standard deviation of these two plots. I found that these two methods have the same level of accuracy.
# 

# ## Q2 - Simulation
# A very common use of Monte Carlo is in simulating experimental data. In simulations, an entire experiment can be reproduced data point by data point, with random numbers being used to model unknowable or changing effects such as the experimental resolution or quantum variations.
# 
# In this question, we will simulate the cosmic ray experiment shown below.
# ![CR-expt.png](attachment:CR-expt.png)
# The experiment comprises 4 detection layers, each of which will produce a signal when a particle traverses the detector, separated by three sheets of copper, which will stop a fraction of muons, allowing a measurement of the muon lifetime to be made.
# 
# You can assume the detector has the following parameters :
# * the efficiency of each of the 4 layers to detect a muon or electron is, from top to bottom : 55%, 60%, 85%, 50%.
# * the probability of a cosmic ray muon to stop in 1cm of copper is $5\times10^{-3}$.
# * electrons are emitted isotropically during decay of a stopped muon.
# * decay electrons have energy 50 MeV and maximum path length of 1.8cm in copper.
# 
# In order to model the initial distribution of cosmic rays, we can assume the anuglar distribution is proportional to ${\rm cos}^2(\theta)$, for zenith angle $\theta$.  The overall normaliation can be taken from the intensity of _vertical_ muons to be 70 $m^{-2}s^{-1}sr^{-1}$. (See PDG review of cosmic rays : http://pdg.lbl.gov/2019/reviews/rpp2019-rev-cosmic-rays.pdf)
# 
# 
# ### 2a)
# Using the model above, write code to simulate each muon that passes nearby the experiment. You will need to generate random numbers with appropriate distributions for the starting point and direction of each muon, and propagate each muon to the detector. You should generate further random numbers to model the stopping and decay process; ie. whether a muon stops in a given layer, and the direction of the decaying electron.
# 
# (Note that for the electron decay, you should generate points that are uniformly distributed on the unit sphere - simply generating two angles between 0 and $2\pi$ will _not_ give the correct distribution!)
# 
# You should discuss the design of your code in the text cell below.

# In[79]:


from mpl_toolkits.mplot3d import Axes3D
#Initial Zenith Angle
def initial_angle(r):
    x=0
    y=1
    while y>x:
        #Angle can be between 0 and pi/2.
        angle=m.floor(m.pi*0.5*np.random.random(1)*10**r)/10**r
        x=(m.cos(angle))**2
        y=np.random.random(1)
    return angle

#Defining a unit sphere distribution.
def unit_sphere():
    x_r=2*np.random.random(1)-1
    y_r=2*np.random.random(1)-1
    z_r=2*np.random.random(1)-1
    norm=1/(x_r**2+y_r**2+z_r**2)**.5
    x_e=(norm*x_r)
    y_e=(norm*y_r)
    z_e=(norm*z_r)
    theta_elec=2*m.atan((x_e**2+y_e**2)**(.5)/z_e)
    phi_elec=m.atan(y_e/x_e)
    return theta_elec, phi_elec, x_e, y_e, z_e

def detector(z,p,x_0,y_0,dx,dy,Z):
    x=x_0+(z-Z)*dx
    y=y_0+(z-Z)*dy
    c=np.random.random(1)
    if c<=p:
        #Location of Detection.
        measured=[x,y,z]
    elif c>p:
        measured=[]
    if x>20 or y>20 or x<0 or y<0:
        measured=[]   
    return measured

def copper(z,p,x_0,y_0,l,dx,dy,theta):
    x=x_0+z*dx
    y=y_0+z*dy
    c=np.random.random(1)
    if c<p:  
        #Location of Decay.
        d=np.random.random(1)*l
        x_decay=float(x+(dx**2)*d)
        y_decay=float(y+(dy**2)*d)
        z_decay=float(z+d*m.cos(theta))
        decay=[x_decay, y_decay, z_decay]
        if x_decay>20 or y_decay>20 or x_decay<0 or y_decay<0:
            decay=[]
    elif c>p:
        decay=[]
    return decay
    
#Set the detectors at z=20,14,6,0
#Set the copper sheets at z=(18-17),(10.5-9.5),(3-2)
#Define path using these conditions for placement of detector and copper.
   
def Path(N):
    counter1=0
    counter2=0
    counter3=0
    counter4=0
    counter_d1=0
    counter_d2=0
    counter_d3=0
    e_counter1=0
    e_counter2=0
    e_counter3=0
    e_counter4=0
    counter_de1=0
    counter_de2=0
    counter_de3=0
    x_i=[]
    y_i=[]
    for i in range(N):
        #Angles.
        theta=float(initial_angle(2))
        phi=float(2*m.pi*np.random.random(1))
        #Initial Position.
        x_0=float(40*np.random.random(1)-10)
        y_0=float(40*np.random.random(1)-10)
        #For plotting the distribution of initial position.
        x_i.append(x_0)
        y_i.append(y_0)
        #Trig functions.
        dx=m.sin(theta)*m.cos(phi)
        dy=m.sin(theta)*m.sin(phi)
        
        #Muon Detections.
        measured1=detector(0,0.55,x_0,y_0,dx,dy,0)
        measured2=detector(6,0.6,x_0,y_0,dx,dy,0)   
        measured3=detector(14,0.85,x_0,y_0,dx,dy,0)
        measured4=detector(20,0.5,x_0,y_0,dx,dy,0) 
        
        #Decays.   
        l=(dx**2+dy**2+1)**0.5
        p=0.005*l
        decay1=copper(2,p,x_0,y_0,l,dx,dy,theta)
        decay2=copper(9.5,p,x_0,y_0,l,dx,dy,theta)
        decay3=copper(17,p,x_0,y_0,l,dx,dy,theta)
        
        #Electron production and decay.
        e_measured1=[]
        e_measured2=[]
        e_measured3=[]
        e_measured4=[]
        decay_e1=[]
        decay_e2=[]
        decay_e3=[]
        e_t, e_p, e_x, e_y, e_z=unit_sphere()
        e_dx=e_x/e_z
        e_dy=e_y/e_z
        #In Question 2c, "... electron that can be detected in an adjacent scintillator layer."
        #So will only code for ADJACENT scintillator layers.
        #So as not to "blur" the results.
        if np.shape(decay1)==(3,):
            x,y,z=decay1
            decay2=[]
            decay3=[]
            measured2=[]
            measured3=[]
            measured4=[]
            ex_i, ey_i, ez_i=decay1
            if e_z>0:
                e_measured1=detector(0,0.55,ex_i,ey_i,e_dx,e_dy,ez_i)
                p_l=1.8*np.random.random()
                p_ly=e_dy**2*p_l
                p_lx=e_dx**2*p_l
                if p_l<((z-2)**2*(e_dx**2+e_dy**2+1))**.5:
                    decay_e1=[x+p_lx,y+p_ly,z+p_l*m.cos(e_t)*m.cos(e_p)]
                    if x+p_lx>20 or x+p_lx<0 or y+p_ly>20 or y+p_ly<0:
                        decay_e=[]
                elif p_l>((z-2)**2*(e_dx**2+e_dy**2+1))**.5:
                    decay_e1=[]
                if np.shape(decay_e1)==(3,):
                    e_measured1=[]
                    
            elif e_z<0:
                e_measured2=detector(6,0.6,ex_i,ey_i,e_dx,e_dy,ez_i)
                p_l=1.8*np.random.random()
                p_ly=e_dy**2*p_l
                p_lx=e_dx**2*p_l
                if p_l<((3-z)**2*(e_dx**2+e_dy**2+1))**.5:
                    decay_e1=[x+p_lx,y+p_ly,z+p_l*m.cos(e_t)*m.cos(e_p)]
                    if x+p_lx>20 or x+p_lx<0 or y+p_ly>20 or y+p_ly<0:
                        decay_e1=[]
                elif p_l>((3-z)**2*(e_dx**2+e_dy**2+1))**.5:
                    decay_e1=[]           
                if np.shape(decay_e1)==(3,):
                    e_measured2=[]
                    
        if np.shape(decay2)==(3,):
            x,y,z=decay2
            decay3=[]
            measured3=[]
            measured4=[]
            ex_i, ey_i, ez_i=decay2
            if e_z>0:
                e_measured2=detector(6,0.6,ex_i,ey_i,e_dx,e_dy,ez_i)
                p_l=1.8*np.random.random()
                p_ly=e_dy**2*p_l
                p_lx=e_dx**2*p_l
                if p_l<((z-9.5)**2*(e_dx**2+e_dy**2+1))**.5:
                    decay_e2=[x+p_lx,y+p_ly,z+p_l*m.cos(e_t)*m.cos(e_p)]
                    if x+p_lx>20 or x+p_lx<0 or y+p_ly>20 or y+p_ly<0:
                        decay_e2=[]
                elif p_l>((z-9.5)**2*(e_dx**2+e_dy**2+1))**.5:
                    decay_e2=[]
                if np.shape(decay_e2)==(3,):
                    e_measured2=[]
                    
            elif e_z<0:
                e_measured3=detector(14,0.85,ex_i,ey_i,e_dx,e_dy,ez_i) 
                p_l=1.8*np.random.random()
                p_ly=e_dy**2*p_l
                p_lx=e_dx**2*p_l
                if p_l<((10.5-z)**2*(e_dx**2+e_dy**2+1))**.5:
                    decay_e2=[x+p_lx,y+p_ly,z+p_l*m.cos(e_t)*m.cos(e_p)]
                    if x+p_lx>20 or x+p_lx<0 or y+p_ly>20 or y+p_ly<0:
                        decay_e2=[]
                elif p_l>((10.5-z)**2*(e_dx**2+e_dy**2+1))**.5:
                    decay_e2=[]
                if np.shape(decay_e2)==(3,):
                    e_measured3=[]
            
        if np.shape(decay3)==(3,):
            x,y,z=decay3
            measured4=[]
            ex_i, ey_i, ez_i=decay3
            if e_z>0:
                e_measured3=detector(14,0.85,ex_i,ey_i,e_dx,e_dy,ez_i)
                p_l=1.8*np.random.random()
                p_ly=e_dy**2*p_l
                p_lx=e_dx**2*p_l
                if p_l<((z-17)**2*(e_dx**2+e_dy**2+1))**.5:
                    decay_e3=[x+p_lx,y+p_ly,z+p_l*m.cos(e_t)*m.cos(e_p)]
                    if x+p_lx>20 or x+p_lx<0 or y+p_ly>20 or y+p_ly<0:
                        decay_e3=[]
                elif p_l>((z-17)**2*(e_dx**2+e_dy**2+1))**.5:
                    decay_e3=[]
                if np.shape(decay_e3)==(3,):
                    e_measured3=[]
                    
            elif e_z<0:    
                e_measured4=detector(20,0.5,ex_i,ey_i,e_dx,e_dy,ez_i)
                p_l=1.8*np.random.random()
                p_ly=e_dy**2*p_l
                p_lx=e_dx**2*p_l
                if p_l<((18-z)**2*(e_dx**2+e_dy**2+1))**.5:
                    decay_e3=[x+p_lx,y+p_ly,z+p_l*m.cos(e_t)*m.cos(e_p)]
                    if x+p_lx>20 or x+p_lx<0 or y+p_ly>20 or y+p_ly<0:
                        decay_e3=[]
                elif p_l>((18-z)**2*(e_dx**2+e_dy**2+1))**.5:
                    decay_e3=[]
                if np.shape(decay_e3)==(3,):
                    e_measured4=[]
        
        #Working out the Counts of each Scintillator.
        if np.shape(measured1)==(3,):
            counter1+=1
        if np.shape(e_measured1)==(3,):
            e_counter1+=1
        if np.shape(measured2)==(3,):
            counter2+=1
        if np.shape(e_measured2)==(3,):
            e_counter2+=1
        if np.shape(measured3)==(3,):
            counter3+=1
        if np.shape(e_measured3)==(3,):
            e_counter3+=1    
        if np.shape(measured4)==(3,):
            counter4+=1
        if np.shape(e_measured4)==(3,):
            e_counter4+=1    
        #Number of Decays in each copper sample.
        if np.shape(decay1)==(3,):
            counter_d1+=1
        if np.shape(decay2)==(3,):
            counter_d2+=1
        if np.shape(decay3)==(3,):
            counter_d3+=1
        if np.shape(decay_e1)==(3,):
            counter_de1+=1
        if np.shape(decay_e2)==(3,):
            counter_de2+=1
        if np.shape(decay_e3)==(3,):
            counter_de3+=1
            
        count_m=[counter1,counter2,counter3,counter4] 
        count_d=[counter_d1,counter_d2,counter_d3]
        count_e=[e_counter1,e_counter2,e_counter3,e_counter4]
        count_de=[counter_de1,counter_de2,counter_de3]
    
    return count_m, count_d, count_e, count_de, x_i, y_i


# # Design of Code:
# 
# The outlining structure of the code is that I have set up a function called Path, in which I completely describe the path of the Muons and Electrons produced for a total of "n" Muons, which I can input when I call the function. I wrote other functions to both produce certain distributions needed to describe the Muon and Electron trajectories, and simulate the probability nature of the scintillator and copper layers.
# 
# To find the incoming angle of each Muon I used the accept reject method to give the zenith (theta) angle between the z-axis and x/y-axis with a probability distritbution of cos^2(theta). To give the phi angle around the x and y-axis, I round generate a number between 0rad and 2*(pi)rad.
# To setup the initial (x,y) coordinates of each Muon I randomly generate numbers between -10cm to 30cm, using both the initial angles and position I was able to propogate the Muon through all stages of the detector.
# 
# To see if the Muon would be detected in a scintillator, layer I wrote a function called "detector" which takes arguments for various initial conditions and returns a set of coordinates which shows where the Muon was detected in the scintillator if it was indeed detected.
# 
# To see if the Muon would decay in a copper layer, I wrote a function called "copper" which takes arguments for various initial conditions and returns a set of coordinates which shows where the Muon decays in the copper layer if it decayed at all.
# 
# For when a Muon decays, in order to produce a set of directions for the electron, I wrote a function called "unit_sphere" which simply returns an (x,y,z) unit vector in its component parts, and a theta and phi which describe polar coordinate angles.
# 
# For the possible decay of an Electron I chose not to use the copper probability since the chance that an electron decays in a copper layer is dependent on its path length which was chosen to be randomly distributed between 0 and 1.8cm. If an electron travelled further then its path length in copper then it instantly decays, if not it leaves the copper. If it did decay, the coordinates of where it decays are given.
# 
# For the detection of electrons, I use the "detector" function to see if it would be detected in the adjacent layers to which it was produced. If it is detected, the coordinates of where it is detected are given.
# 
# To count the respective detections and decays, I setup 4 sets of counters at the start equal to zero. Then if any of the coordinates of where things are detected/decay ==(3,) then the respective counter gets +1 added. The idea behind this being that it either a Moun or Electron are not detected or do not decay then they have shape ==(0,) and the respective counter will not be added to.
# 
# These counters are then returned, along with a list of all the initial positions.
# 
# 
# ##################################################################################################################
# ##################################################################################################################
# 
# Assumptions of my code: (Given within the code as well).
# 
# The positions and thickness of my scintillator/copper layers.
# 
# Scintillators:  z=20, z=14, z=6, z=0. Each with thickness 0cm.
# 
# Coppers: z=18-17, z=10.5-9.5, z=3-2. With given thickness 1cm.
# 
# Electrons are only able to be detected by adjacent scintillators.
# 
# Path length of Electrons is has a uniform probability of being between 0-1.8cm.
# 
# Muons are produced at the top layer of the scintillator.
# 
# That the probability a Muon decays in L(cm) of copper is equal to  0.05*L.
# 
# These are assumptions were made either for the ease of coding or because I was mislead by the question, not because they were thought to be true.
# 

# ### 2b)
# In the next cell you should validate your code.  The aim is to test separate parts of the code individually, to ensure the expected distributions or behaviour are produced in each case.

# In[82]:


#Testing the distribution of starting angle.
a=[initial_angle(3) for i in range(100000)]
plt.hist(a, 100, histtype='step', label="Iterated Distribution")
plt.title("Testing the distribution of the Accept/Reject method in 2a.", fontsize=12)
plt.xlabel("Value of Angle (rad)", fontsize=13)
plt.ylabel("Frequency of Angle", fontsize=13)
plt.legend()
plt.show()

#Testing the distribtion of electron emission.
figure=plt.figure()
gph=figure.add_subplot(111, projection='3d')
pe=[]
te=[]
xe=[]
ye=[]
ze=[]
for i in range(10000):
    q,w,e,r,t=unit_sphere()
    te.append(q)
    pe.append(w)
    xe.append(e)
    ye.append(r)
    ze.append(t)
gph.scatter3D(xe, ye, ze, zdir='z', s=1, alpha=0.15)
plt.title("The distribution of the normalised paths the electrons take after a muon decay at the origin", fontsize=11)
plt.show()
print("Wait... next thing is about 20 seconds")

#Testing both the distribution of initial position, and counts/decays at each detector or in each copper respectively.
a=Path(100000)
plt.scatter(a[4],a[5], s=0.2, label="Initial Positions")
plt.title("Testing the distribution of initial position at the first detector.", fontsize=11)
plt.xlabel("Initial X position (cm)", fontsize=13)
plt.ylabel("Initial Y position (cm)", fontsize=13)
plt.legend()
plt.show()
print("")
print("-------------------------------------------------------------------------------------------------------")
print("Number of Detection/Decay Counts for 100000 Muons Entering the Detector.")
print("")
print("Muons Detected at Scintillator 1: ", a[0][0], "   Number of Electrons Detected at Scintillator 1: ", a[2][0])
print("Muons Detected at Scintillator 2: ", a[0][1], "   Number of Electrons Detected at Scintillator 2: ", a[2][1])
print("Muons Detected at Scintillator 3: ", a[0][2], "   Number of Electrons Detected at Scintillator 3: ", a[2][2])
print("Muons Detected at Scintillator 4: ", a[0][3], "   Number of Electrons Detected at Scintillator 4: ", a[2][3])
print("")
print("Number of Muon Decays in Copper 1: ", a[1][0], "    Number of Electrons Decays in Copper 1: ", a[3][0])
print("Number of Muon Decays in Copper 2: ", a[1][1], "    Number of Electrons Decays in Copper 2: ", a[3][1])
print("Number of Muon Decays in Copper 3: ", a[1][2], "    Number of Electrons Decays in Copper 3: ", a[3][2])
print("-------------------------------------------------------------------------------------------------------")


# # Validation:
# 
# The distribution for the initial theta angle, between the z and x/y-axes, is shown to have a probability distribution of cos^2(theta) between 0 and pi/2. 
# 
# The distribution of normalised paths is also distributed correctly, as it shows a unit sphere.
# 
# The distribution of initial positions a Muon may have is uniform which is also expected.
# 
# \
# 
# Number of Muon detections:
# 
# For each scintillator the number of detections should be, detector efficiency * Number of Muons passing through that scintillator plate. N.B. since most of the Muons are produced far from the detector the number will be lower then simply (Number of Muons)*(efficiency of detector)
# 
# 
# Due to the design of the code, the number of Muons through each scintillator plate should be roughly the same for each plate. So the number of detections in each should be proportional to the scintillator efficiency.
# 
# 
# 

# ### 2c)
# Now, use your simulation to estimate :
# * The total rate of muons that pass through the detector.
# * The fraction of those muons which are registered in 1, 2, 3 or 4 scintillator planes of the detector. 
# * The fraction of those muons which decay and produce an electron that can be detected in an adjacent scintillator layer.

# In[83]:


n=10000
A=Path(n)

# Intensity of Vertical Muons is 70 m−2s−1sr−1.
# Through our detector given the angle of the incoming muons and the area of detector
# the intensity is 13.817 Muons per second.
# Given the fraction of Muons detected at the top is 0.135, and the fraction of Muons detected at the bottom
# is 0.115. We can assume that just under 100% of the Muons that go through the top also come out the bottom.
# This assumption is made using the respective detector efficiencies.
# So estimated rate of Muons is 13 per second.
print("The Total Rate of Muons that pass through the Detector: ", 13,"per second")
print("")
print("")
print("Fraction of these Muons going through the Detector which are Registered in Scintillator 1: ",A[0][0]/(.25*n))
print("Fraction of these Muons going through the Detector which are Registered in Scintillator 2: ",A[0][1]/(.25*n))
print("Fraction of these Muons going through the Detector which are Registered in Scintillator 3: ",A[0][2]/(.25*n))
print("Fraction of these Muons going through the Detector which are Registered in Scintillator 4: ",A[0][3]/(.25*n))
print("")
print("")
print("Fraction of these Muons which Decay and Produce an Electron: ",(A[1][0]+A[1][1]+A[1][2])/(.25*n))
print("Fraction of Electrons which are Detected in an Adjacent Scintillator: ",(A[2][0]+A[2][1]+A[2][2]+A[2][3])/(A[1][0]+A[1][1]+A[1][2]))
print("")
print("Therefore the fraction of these Muons which Decay and Produce an Electron which can")
print("be is Detected in an Adjacent Scintillator layer is: ",(A[2][0]+A[2][1]+A[2][2]+A[2][3])/(.25*n))


# # Results:
# 
# The estimation for the total rate of Muons that pass through the detector was 13 per second. This estimation uses the simulation results to see what fraction of Muons make it from the top scintillator layer to the bottom. The estimation also required integrating the intensity of vertical Muons over the angle distribution and total area of the detector.
# 
# The fraction of these Muons which are registered in scintillator 1,2,3, and 4, should, in theory, be equal to the scintillator efficiencies. But in our psuedo-experiment, because the Muons are only generated from a square centered at the centre of of detector with a scale factor size of 2, there will be a drop off in Muons detected as we go down the scintillator levels. Since the Muons were chosen to be generated from a square with scale factor 2 to that of our detector, I had to divide the total number of Muons which went all the way through the detector by 4. This is where the factor of 0.25 comes from.
# 
# The fraction of Muons which decay and produce an electron which can be detected by an adjacent layer was broken down into its two consituent parts.
# 
# 1) Fraction of Muons which decay.
# 
# 2) Fraction of Electrons detected in an adjacent scintillator layer.
# 
# This was done to better understand where the end fraction comes from.
# 
# \\
# 

# This question is well suited to extensions. For example, a negative muon stopping in the Copper may be "captured" by an atomic nucleus, which modifies its lifetime (to ~0.164 $\mu s$). Positive muons are not captured and hence their lifetime is unaffected. You can simulate this, to estimate the expected distribution of muon decay times.  (An even more detailed simulation could include muons that stop in scintillator...)
# 
# Feel free to discuss possible extensions with your demonstrator and/or the unit director !

# ## Q3 - Statistical Analysis
# 
# In this question, we will explore the use of Monte Carlo methods to understand experiment outcomes.
# 
# Standard experimental error analysis frequently uses the assumption that uncertainties are normally distributed. The interpretation of a result quoted as $\mu \pm \sigma$ is taken that the true value lies within the range [$(\mu - \sigma$),$(\mu + \sigma)$] with a certain probability (usually 68%). However, it is not hard to find cases where these assumptions break. A classic example occurs when measuring a parameter that is close to a physical boundary - the standard error treatment may result in a range of values that includes the non-physical region.
# 
# A more sophisticated approach is to treat the measurement process as an inverse problem, ie. the inference of model parameters from experimental measurements. (For example, estimation of G from observations of planetary motion). Given a model, we can use Monte Carlo techniques to generate ensembles of "pseudo-experiments", and build up distributions of experimental outcomes for a given set of model parameters. Then it is straightforward to find the range of model parameters that cover an actual experimental observation, within a specified probability.
# 
# ### 3a)
# 
# A "counting experiment" is performed at a collider, to search for the existence of a hypothesised new particle.  The experiment consists of counting the number of events that meet certain criteria. Events can be produced either by the hypothetical signal process, or by known background processes. However, an individual event cannot be ascribed to be signal or background - the only measurable quantity is the _total_ number of events.
# 
# Both signal and background processes produce a number of events that is Poisson distributed. The mean number of background events has been estimated to be $4.8 \pm 0.5$.  The mean number of signal events is given by $L \sigma$, where the integrated luminosity $L=10 nb^{-1}$, and $\sigma$ is the (unknown) cross section of the signal process. The number of events observed in the counting experiment is 6.
# 
# You should write a Monte Carlo programme that will calculate the upper limit on the signal cross section that is compatible with the observation at 95% confidence level.
# 
# You will need to generate pseudo-experiments for a range of different signal cross sections. For each pseudo-experiment, generate random numbers to model the Gaussian uncertainty on the background prediction, and the Poisson variation in both the background and signal production. Ensure that the number of pseudo-experiments are sufficient to measure the experimental distribution for each cross section, and in particular the fraction of the distribution that is _greater_ than the measured value (the confidence level).
# 
# How would you incorporate additional uncertainties?  For example, if the uncertainty on the luminosity is 5%, or the efficiency to identify signal events is estimated to be $0.7 \pm 0.1$ ?

# In[73]:


#L=10nb**(-1). Which is equivalent to 10**36 m**(-2). But we will use L=10 nb**(-1)

all_counts_mean=[]
N=200
n=401
k=2000
for i in range(n):
    counts=[]
    L=np.random.normal(10,0.5)
    signal_efficiency=np.random.normal(0.7,0.1)
    B_lam=np.random.normal(4.8,0.5)
    for j in range(N):
        B=np.random.poisson(B_lam)
        S=np.random.poisson(L*(i/k))
        if np.random.random()<signal_efficiency:
            counts.append(S+B)
        elif np.random.random()>signal_efficiency:
            counts.append(B)
    vals, freq=np.unique(counts,return_counts=True)
    all_counts_mean.append(np.mean(counts))
    if i==0 or i==80 or i==160 or i==240 or i==320 or i==400:
        plt.plot(vals,freq)
        print("The mean number of counts at a sigma value of",i/k,"is: ",np.mean(counts))
        print("The signal efficiency is: ",signal_efficiency)
        print("The luminosity value is: ",L)
        print("The mean background count is: ", B_lam)
        print("")
print("The distribution for these sigma values are plotted below.")
plt.xlabel("Number of counts due to the background events.")
plt.ylabel("Frequency of the number of counts.")
plt.title("A plot to show the frequency of number of background counts for some values of sigma.")
plt.show()
print("This graph above shows the different total distributions for different values")
print("of lambda for the signal distribution. When L=10+/-0.5 and the efficiency")
print("to identify signal events is estimated to be 0.7+/-0.1.")
print("")

all_sigma=[i/k for i in range(n)]
plt.plot(all_sigma,all_counts_mean, label="The mean counts against lambda value")
#95% confidence interval is lambda +/- 1.96*root(lambda/n)
upper_limit=6+1.96*np.sqrt(6/N)
line=np.full(len(all_sigma),upper_limit)
plt.plot(all_sigma,line, label="y=upper-limit of 95% confidence interval")

m,c=np.polyfit(all_sigma,all_counts_mean,1)
for i in range(len(all_sigma)):
    if upper_limit<(m*all_sigma[i]+c):
        break
plt.plot(all_sigma,np.multiply(m,all_sigma)+c, label="Line of best fit")        
# x is how far along the line the intercept is a fraction
# of the length of the line between the two points it falls between.
# 'cept' is the upperbound for the sigma value.
x=(upper_limit-all_counts_mean[i-1])/(all_counts_mean[i]-all_counts_mean[i-1])
cept=x*(all_sigma[i]-all_sigma[i-1])+all_sigma[i-1]
intercept=np.full(len(all_sigma),cept)
plt.plot(intercept,all_counts_mean, label="x=corresponding value for lambda")
plt.legend(fontsize=8)
plt.xlabel("Values of lambda for the signal event distribution.")
plt.ylabel("Number of mean counts.")
plt.title("Finding an upper-limit of the lambda of the signal distribution.")
plt.show()
print("The upper-limit of the 95% confidence interval around 6 counts: ", upper_limit)
print("The corresponding value for sigma which gives this upper limit: ",cept)


# # Code Explanation:
# 
# In my first for loop, I cycle through different values of sigma. The values of sigma are calculated as i/k. 
# i is cycled from 0 to 401 and k was chosen to be 2000. Then, for each value of sigma I calculate a new L value, a new signal detection efficiency, this was since it was suggested in the code but I also calculate a new mean value for the number of background counts. I calculate L, detection efficiency, and the background lambda as gaussian distributions with a standard deviation equal to the +/- values.
# 
# Then for each sigma I run the experiment 2000 times (in a 2nd loop), this was in an effort to make the 95% confidence interval lower.
# I then plot the poisson distributions for certain values of sigma so as to not clutter the graph, I also print the raw results to see how the code is working in terms of numbers generated.
# 
# After these two loops, having appended the necessary values in lists which can be plotted. I calculate the upper limit to the 95% confidence interval around 6 counts. Then having found the line of best fit of the sigma values against the mean number of counts, I compute where the line of best fit crosses the upper limit value to give the value of sigma which gives the upper bound on the confidence interval.
# 
# I then use this value of sigma as my upper limit on sigma.
# 
# 
# # Results:
# 
# The upper limit on the 95% confidence interval is calculated to be 6.3395 at an N value of 200. Given the inaccuracies of the pseudo-experiment, the value of sigma which my code finds to correspond with this upper limit is around 0.18.
# 
# In a perfect scenario, with a Background count of 4.8 and perfect signal detection and an L value of 10. A sigma value of 0.15 would lead to an expected value of counts of 6.3. This seems okay for an upper limit, but is ambitious.
# 
# It's ambitious because if the experiment is done once. Since signal detection efficiency is 0.7+/-0.1, L is 10+/-0.5, and mean of background counts is 4.8+/-0.5. You could argue that the upper limit should be when:
# 
# 1) Signal detection efficiency is 0.6.
# 
# 2) L is 9.5.
# 
# 3) Bckground count mean is 4.3.
# 
# Where for an expected number of counts to be 6. A sigma of 0.298 is required. And that's without the 95% confidence limit, so it should be even higher.
# 
# The ambitious result comes from the fact that the experiment is run many times, giving a closer confidence interval, and the uncertain variables are not taken to be their lowest value.
# 
# However my value would be representative if the experiment in real-life was done multiple times.

# In[ ]:




