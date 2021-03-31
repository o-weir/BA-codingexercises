#!/usr/bin/env python
# coding: utf-8

# # Assignment 2 - Differential Equations
# 
# Solving partial differential equations is crucial to a huge variety of physical problems encountered in science and engineering. There are many different numerical techniques available, all with their own advantages and disadvantages, and often specific problems are best solved with very specific algorithms.
# 
# You will have learnt about Euler and Runge-Kutta methods in 2nd year lectures, and you should have explored the class of problem that can be solved with numerical integration in exercises.  In this assignment, we will cover more complex classes of problem - described below.
# 
# 
# ## Initial value problems
# 
# In this class of problem, the state of a system is fully described by an ordinary differential equation together with an initial condition.  For example, the motion of a body under gravity, with initial conditions given by the position and momentum of the body at a particular point in time.  The soluiton (ie. position and momentum at an arbitrary time in the future) can then be found by integration.  You should have encountered the use of numerical integration in solving such problems in the 2nd year course.
# 
# ## Boundary value problems
# 
# Boundary value problems differ in that the conditions are specified on a set of boundaries, rather than at just one extreme.  For example, the electric field between a pair of capacitor plates at fixed potential, as discussed in the problem below.
# 
# There are several numerical approaches for solving boundary value problems, for example :
# 
# ### Shooting Method
# 
# In this method, the boundary value problem is reduced to an initial value problem, which is solved numerically for different parameter choices. A solution is found when a set of parameters give the desired boundary conditions.  For example, finding a rocket trajectory which joins two specified points in space.  The boundary conditions are the specified points, and the initial momentum is a parameter that may be varied until a solution is found.  (This should sound familiar!)
# 
# ### Finite Difference Methods
# 
# In this class of method, the differential equation is evaluated at discrete points in space and time, and derivatives are approximated by finite differences.  The Euler and Runga-Kutta methods are simple examples.  These methods typically involve iteration on the set of finite values until a solution is found.
# 
# ### Relaxation
# 
# This is a common technique used to solve time-independent boundary condition problems.  An initial guess at the solution is supplied, and then allow to "relax" by iterating towards the final solution.  Conceptually this is is the same as the time-dependent problem of letting the system reach equilibrium from some arbitrary initial state.
# 
# The steps for implementing a relaxation method are :
# 1. Define a (normally regular) spatial grid covering the region of interest including points (or “nodes”) on the boundaries
# 2. Impose the boundary conditions by fixing the nodes on the boundaries to the relevant values
# 3. Set all non-boundary nodes to an initial guess
# 4. Write down finite difference equations
# 5. Pick a convergence criterion
# 6. Iterate the equations at each node until the solution converges
# 
# Care must be taken to choose the form of the equations and iteration method to ensure stability and efficiency.

# In[ ]:





# ## Q1 - The Poisson Equation
# 
# Consider the example of the Poisson equation $(\nabla^2V = −\rho)$ in one dimension. The grid of nodes in this case can be taken as a series of $n$ equally spaced points $x_i$ with a spacing $\Delta x = h$. The Taylor expansion of $V$ around the point $x_i$ is :
# 
# $$ V(x) = V(x_i) + \delta x \frac{dV(x_i)}{dx} + \delta x^2 \frac{d^2V(x_i)}{dx^2} + ...$$
# 
# so adding the values at $\delta x = \pm h$ (i.e. at $x_n \pm 1$) gives :
# 
# $$ V(x_{i−1}) + V(x_{i+1}) = 2V(x_i) + h^2 \frac{d^2V(x_i)}{dx^2} $$
# 
# which can be rearranged to give Equation 1 :
# 
# $$ \frac{d^2V(x_i)}{dx^2} = \frac{V(x_{i−1}) + V(x_{i+1}) − 2V(x_i)}{h^2}  $$
# 
# This is the standard finite difference representation for the second derivative.
# 
# Generalising this equation to 2D in the Poisson equation, and rearranging, gives Equation 2, that can be used to iterate the value at each node:
# 
# $$ V(x_i,y_j)= \frac{1}{4} (V(x_{i−1},y_j)+V(x_{i+1},y_j)+V(x_i,y_{j−1})+V(x_i,y_{j+1}))+ \frac{\rho(x_i,y_j)h^2}{4} $$
# 
# In the absence of any sources ($\nabla^2 V=0$, i.e. the Laplace equation) each node is simply the average of its four closest neighbours.
# 
# This equation can be solved in a number of ways. One option is to calculate a new value for each node based on the previous values for each of the neighbour nodes, requiring two complete copies of the grid. This is called the Jacobi method. A second option is to update the values on the grid continually, so each node is updated using partially old and partially new values. This is the Gauss-Seidel method.

# ## 1a) 
# Write a function to solve Laplace’s equation in two dimensions for the potential V. You should use the finite-difference representation above (with $\rho=0$) and iterate using either the Jacobi or Gauss-Seidel method. You will need to choose and apply a convergence condition e.g. no node value changes by more than X% between successive iterations.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt


class Base:
    
    def __init__(self, dx=0.1, l=10, v=5):   
        self.dx=dx
        self.Pot=v
        self.pixel=int(l/dx)
        self.v=np.zeros((self.pixel, self.pixel))
        self.bound=np.zeros((self.pixel, self.pixel))
    
    
    def Jacob(self, v, error):

        v_new=np.zeros((self.pixel, self.pixel))
        v_old=np.zeros((self.pixel, self.pixel))
        v_new[:]=v_old[:]=self.v
        
        thing=False
        while thing==False:   
            for i in range(1, self.pixel-1):
                for j in range(1, self.pixel-1):
                    if self.bound[i][j]!=0:
                        continue
                    v_new[i][j]=(1/4)*(v_old[i-1][j]+v_old[i][j-1]+v_old[i][j+1]+v_old[i+1][j])
            if np.allclose(v_old, v_new, atol=error) == True:
                break
            v_old[:]=v_new[:]
        
        return v_new        
        
        
    def Graph(self, Ttl, v, colour):
        plt.imshow(v)
        plt.colorbar(label = 'Voltage')
        plt.set_cmap(colour)
        plt.title(Ttl, fontsize=18)
        plt.xlabel('X Coordinate', fontsize=13)
        plt.ylabel('Y Coordinate', fontsize=13)
        plt.show()
        
  
class Parallel_Plate(Base):
    
    def Bound(self, b, d):
        for i in range(int(self.pixel*b)):
            self.v[int(self.pixel*(0.5+(d/2)))][i+int(self.pixel*((1-b)/2))]=-self.Pot
            self.v[int(self.pixel*(0.5-(d/2)))][i+int(self.pixel*((1-b)/2))]=self.Pot
        self.bound[:]=self.v[:]

class Singularity(Base):
    
    def Bound(self, x, y):
        self.v[int(y*self.pixel), int(x*self.pixel)]=self.Pot
        self.bound[:]=self.v[:]

class Disc(Base):
    
    def Bound(self, cent_x, cent_y, r):
        v=np.zeros_like(self.v)
        H,K=np.meshgrid(np.arange(v.shape[0]),np.arange(v.shape[1]))
        D=np.sqrt((H-(cent_y*self.pixel))**2+(K-(cent_x*self.pixel))**2)
        v[np.where(D<(r*self.pixel))]=self.Pot
        self.v[:]=self.bound[:]=v[:] 
        
        
Input='0'
while Input!='Quit':
    print("Don't forget your capitals!!")
    Input=input('Enter a choice of, "Parallel Plate", "Singularity", "Disc" or "Quit", to quit: ')   
    if Input=="Parallel Plate":
        print('You have entered: ', '"',Input,'"')
        Input=Parallel_Plate(0.1, 10, 5)
        Input1='0'
        Input1=input("Type a value of distance separating the parallel plates as a fraction of the width of the box encapsulating them, eg. 0.2:  ")
        print("-- This may take a while --")
        Input.Graph('Parallel Plate', Input.Jacob(Input.Bound(.8, float(Input1)), 1e-3), 'seismic')
        
    elif Input=="Singularity":
        print('You have entered: ', '"',Input,'" -- ',  'This may take a while...')
        Input=Singularity(0.1, 5, 5)
        Input.Graph('Singularity', Input.Jacob(Input.Bound(.5, .5), 1e-3), 'afmhot')
            
    elif Input=="Disc":
        print('You have entered: ', '"',Input,'"')
        Input=Disc(0.1, 10, 5)
        Input2='0'
        Input2=input("Type a value of radius of Disc as a fraction of the width of the box encapsulating them, eg. 0.2:  ")
        print("-- This may take a while --")
        Input.Graph('Disc', Input.Jacob(Input.Bound(.5, .5, float(Input2)), 1e-3), 'afmhot')


# I decided to write this code by setting up a set of classes, on which I can write in subclasses describing different scenarios.
# The properties of the Base case was defined by the __init__ function. I also wrote functions to perform the Jacobian iteration and graph the results.
# The Base case used the terms "dx" and "L" to define both the pixel resolution and size of axis in the situation. If the ratio between them came much larger then 100 for L/dx. Then the computer started taking very long to compute the separate cases.
# For the "Parallel Plate" case, I defined a breadth of the plates and a distance between them in their bound functions, "b" and "d". Then I wrote code using these two variables and variables from the base case to define the boundaries as two plates, both a distance d/2 from the centre, one of the plates then has +v, whilst the other has -v.
# For the "Disc" case, I defined the x ("cent_x") and y ("cent_y") coordinates for the centre of the Disc, and the radius ("r") of the disc, I then used these variables and variables from the base case to define the boundaries as a circle with potential v, all other points were then iterated using this boundary condition.
# For the "Singularity" case, I defined the x ("x") and y ("y") coordinates for the position of the Singularity. I then gave the Singularity a potentail v, and iterated the other pixels using this boundary condtion.
# 
# 
# 
# 
# 

# Verify your function by checking it works in a simple, known case. Compare the solution found with the analytical solution and quantify the differences. Use this to investigate the sensitivity of your solution to the choice of grid density and convergence condition.

# In[33]:


class test(Base):
    
    def Bound(self):
        self.v[:]=np.zeros((self.pixel,self.pixel))
        for i in range(0, self.pixel):
            self.v[i][0]=0
            self.v[i][self.pixel-1]=self.pixel
            self.v[0][i]=i
            self.v[self.pixel-1][i]=i
        self.bound[:]=self.v[:]
      
    def analytical_sol(self):
        for i in range(0, self.pixel):
            for j in range(0, self.pixel):
                self.v[i][j]=j
        return self.v
    
    
      
p=test(0.1, 5)
print("                                        --Please wait a moment--")
p.Graph("Our solution", p.Jacob(p.Bound(), 1e-20), 'seismic')

print("Above is a graph showing the Jacobian iteration of V(x,y)=x from the boundary points (which were the edges of the box")
print("i.e. where x, and or y=0,L) to the rest of the V(x,y) points inside the box.")
print("Looking at the graph we get what we would expect.")

p.Graph("Analytical solution", p.analytical_sol(), 'seismic')

print("Above is a graph where V(x,y)=x is initially plotted for every of x,y without the use of any iteration.")


# As we can see from the 2 graphs the Jacobian method for finite difference iteration gives the exact relationship very closely. You may notice that at x~25 that the two graphs differ. It seems my solution gives smaller values then it's meant to for x<25, and larger values then it is meant to for x>25.

# ## 1b)
# Now use your function to calculate the potential and electric field within and around a parallel plate capacitor comprising a pair of plates of length a, separation d. Demonstrate that the field configuration approaches the “infinite” plate solution (E = V/d between plates, E = 0 elsewhere) as the ratio of  becomes large.

# In[4]:


Input='0'
while Input!='Quit':
    Input=input('Enter a choice of, "Parallel Plate", "Singularity", "Disc" or "Quit", to quit: ')
    if Input=="Parallel Plate":
        print('You have entered: ', '"',Input,'"')
        Input=Parallel_Plate(0.1, 10, 5)
        Input1='0'
        Input1=input("Type a value of the distance separating the two plates as a fraction of the width of the box encapsulating them, eg. 0.2:  ")
        Input2='0'
        Input2=input("Type a value for the length of the plates as a fraction of the length of the box which encapsulates them, eg. 0.9:   ")
        print("-- This may take a while --")
        Input.Graph('Parallel Plate', Input.Jacob(Input.Bound(float(Input2), float(Input1)), 1e-3), 'seismic')
        
    elif Input=="Singularity":
        print('You have entered: ', '"',Input,'" -- ',  'This may take a while...')
        Input3='0'
        Input3=input("Give the x coordinate of the Singularity on an axis going from 0 to 1:  ")
        Input4='0'
        Input4=input("Give the y coordinate of the Singularity on an axis going from 0 to 1:  ")
        Input=Singularity(0.1, 3, 5)
        Input.Graph('Singularity', Input.Jacob(Input.Bound(float(Input3), float(Input4)), 1e-3), 'afmhot')
            
    elif Input=="Disc":
        print('You have entered: ', '"',Input,'"')
        Input8='0'
        Input8=input("Give the resolution, the higher the resolution the longer this will take. In order to see how you can use a Disc to   model a singularity, I suggest resolution 200 with a radius of 0.01:   ")
        Input=Disc(1, float(Input8), 5)
        Input7='0'
        Input7=input("Type a value of radius of Disc as a fraction of the width of the box encapsulating them, eg. 0.2:  ")
        print("-- This may take a while --")
        Input.Graph('Disc', Input.Jacob(Input.Bound(0.5, 0.5, float(Input7)), 1e-3), 'afmhot')


# Parallel Plates:
# When the distance between the plates is very small the field configuration becomes very close to the "infinite plate solution", the range at which this approximation becomes good is dependent on both the length of the two plates, the distance between them, and the voltage difference between the plates.
# Because in my graphs, the colour gradient is the same whatever the potentail difference, it is hard to see the impact that the voltage difference has on the approximation.
# I would argue that the approximation becomes bad when the ratio of d/b (distance separating the plates over the width of the plates) goes higher then 0.9/1.
# 
# Singularity:
# The problem with iterating the singularity ("point charge") via the finite differences method stems from the fact that the singularity is just one point. So the singularity may only effect the potential nodes next to it directly. And although at a high resolution it can't be noticed very well. If you "zoom" into the singularity you notice that the field pattern around the singularity is very blocky and not an accurate representation of real point charge fields.
# One is much better off on a high resolution using a disc of small radius.
# 
# Disc:
# As mentioned, the Disc can be used to model the Singularity field very well if you have a high resloution with a very small disc. Then the Disc will look similar to a Singularity in a low resolution but will have a more accurate representation of the field that a point charge would have in real life. This is due to the fact that the "Point Charge" will now be made up of multiple V(x,y) nodes.
# To see how the Disc may model a singularity well I suggest a resolution of 200 and a radius input of r<0.01

# # Q2 - The Diffusion Equation
# 
# Solving the diffusion equation 
# 
# $$\alpha \nabla^2 \phi = \frac{\partial \phi}{\partial t}$$
# 
# is mathematically similar to solving the Poisson equation. The technique will be to start from known initial conditions and use finite difference equations to propagate the node values forwards in time (subject to any boundary conditions).
# 
# A first try using Equation 1 above gives the finite difference form:
# 
# $$\frac{\phi′(x_i) − \phi(x_i)}{\delta t} = \frac{\alpha}{h^2} [\phi (x_{i−1}) + \phi(x_{i+1}) − 2\phi(x_i)]$$
# 
# Here the values, $\phi$, at three neighbouring points at a time t are used to evaluate the value $\phi`$ at the next time step, $t + \delta t$. This is known as a forward-time or explicit method. Unfortunately, this methood is known to be unstable for certain choices of $h$ and $\delta t$.
# 
# A stable alternative is obtained by using the equivalent backward-time or implicit equation:
# 
# $$\frac{\phi'(x_i) - \phi(x_i)}{\delta t} = \frac{\alpha}{h^2} [\phi'(x_{i-1}) + \phi'(x_{i+1}) -  2\phi'(x_i)] $$
# 
# Now the spatial derivative on the right-hand side needs to be evaluated at $t + \delta t$, which may appear problematic as the $\phi(x)$ values are known while the updated $\phi′(x)$ values are not. Fortunately Equation 3 can be written explicitly in matrix form and solved using the methods explored in Assignment 1.
# 

# ## 2a)
# An iron poker of length 50 cm is initially at a temperature of 20 C. At time t = 0, one end is thrust into a furnace at 1000 C and the other end is held in an ice bath at 0 C. Ignoring heat losses along the length of the poker, calculate the temperature distribution along it as a function of time. You may take the thermal conductivity of iron to be a constant 59 W/m/K, its specific heat as 450 J/kg/K and its density as 7,900 kg/m3.
# 
# Your solution should apply the implicit finite difference method above. It is also recommended that you use an appropriate linear algebra routine from numpy/scipy. You should find ways to verify your results, and quantify the uncertainties due to the method. Discuss your results in the text box below.

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import scipy.linalg
import math

#Defining the Rod
class Rod():
    
    def __init__(self, n_t, n_x):
        
        self.F=(59/450)*7900
        self.t=0.00008
        self.L=0.5
        self.N_t=n_t
        self.N_x=n_x
        self.T_hot=1273
        self.T_cold=273
        self.dx=self.L/self.N_x
        self.dt=self.t/self.N_t 

        self.t_axis=np.arange(0, self.t+self.dt, self.dt)
        self.x=np.arange(0, self.L+self.dx, self.dx)
        self.T_old=np.zeros(self.N_x+1)
        self.T_new=np.zeros(self.N_x+1)

#U=b*A where b is the heat values along the x direction at present time, A are a matrix of coefficients, and U
#are the heat values along the x direction at the next time step.
#Therefore U is what we want to anayltically solve for.

        self.A=np.zeros((self.N_x+1, self.N_x+1))
        self.b=np.zeros(self.N_x+1)
        self.S=(self.F*self.dt/(self.dx)**2)
        for i in range(1, self.N_x):
            self.T_old[i]=293
        self.T_old[0]=self.T_hot
        self.T_old[self.N_x]=self.T_cold
        for i in range(1, self.N_x):
            self.A[i][i-1]=-self.S
            self.A[i][i]=(1+2*self.S)
            self.A[i][i+1]=-self.S
            self.A[self.N_x][self.N_x]=1
            self.A[0][0]=1    
        
    def Jacobi_Euler(self):
        for i in range(1, self.N_t):
#This plot function plots the rod's temperature for every position node at the current iteration time step.
#For each newer iterated time step the colour of the plotted line becomes a darker shade of blue.
            plt.plot(self.x, self.T_old, color=plt.cm.Blues(100+10*i))
            for j in range(1, self.N_x):
                self.b[j]=self.T_old[j]
            self.b[0]=self.T_hot
            self.b[self.N_x]=self.T_cold
            self.T_new[:]=scipy.linalg.solve(self.A, self.b)[:]
            self.T_old[:]=self.T_new[:]       
            
    def Graphing(self):      
        plt.title("Temperature along the rod at different times", fontsize=14)
        plt.xlabel('X Coordinate (cm)', fontsize=15)
        plt.ylabel('Heat, K', fontsize=15)
        plt.show()
    
D=Rod(25, 50)
D.Jacobi_Euler()
D.Graphing()
print("Every different line is the rod at a different time stamp. The darker the shade of blue the further evolved")
print("the rod is.")


# The code used the Class system as in Question 1. Where the Rod is now a separate class. I defined a Graphing function which plotted axis labels and titles to the plot that is made within the iteration. I also defined a backwards Euler function to iterate along the rod at each new timestep. 
# In my code I also re-defined the system of equtions into a set of matrices as described in the code itself so that it could then be solved through linear algebra. This is done so that in the iteration the new temperature for each x coordinate along the rod can be found before the iteration goes onto the next time step.
# 
# I was not able to get the Fourier series to work very well, and therefore do not have an analytical solution to compare my results too. However the rod is behaving exactly as expected, where the final state is a linear heat distribution from 1273K to 273K.

# ## 2b)
# Now repeat the calculation, but assume the far end of the poker from the furnace is no longer held at 0 C, and experiences no heat loss. You will need to modify your code to achieve this, and you should discuss the motivation for your changes below.

# In[6]:


import numpy as np
import matplotlib.pyplot as plt

#Defining the Rod

class Rod_2b():
    
    def __init__(self, t, n_t, n_x):

        self.F=(59/450)*7900
        self.t=t/10000
        self.L=0.5
        self.N_t=n_t
        self.N_x=n_x
        self.T_hot=1273
        self.T_cold=293
        self.dx=self.L/self.N_x
        self.dt=self.t/self.N_t 

        self.t_axis=np.arange(0, self.t+self.dx, self.dt)
        self.x=np.arange(0, self.L+self.dx, self.dx)
        self.T_old=np.zeros(self.N_x+1)
        self.T_new=np.zeros(self.N_x+1)



#U=b*A where b is the heat values along the x direction at present time, A are a matrix of coefficients, and U
#are the heat values along the x direction at the next time step.

        self.A=np.zeros((self.N_x+1, self.N_x+1))
        self.b=np.zeros(self.N_x+1)
        self.S=(self.F*self.dt/(self.dx)**2)
        for i in range(1, self.N_x):
            self.T_old[i]=293
        self.T_old[0]=self.T_hot
        self.T_old[self.N_x]=self.T_cold
        for i in range(1, self.N_x):
            self.A[i][i-1]=-self.S
            self.A[i][i]=(1+2*self.S)
            self.A[i][i+1]=-self.S
        self.A[self.N_x][self.N_x]=2*self.S+1
        self.A[self.N_x][self.N_x-1]=-2*self.S
        self.A[0][0]=1
        print(self.S)
        

    def Jacobi_Euler(self):
        for i in range(1, self.N_t):
            plt.plot(self.x, self.T_old, color=plt.cm.Blues(80+15*i))
            for j in range(1, self.N_x):
                self.b[j]=self.T_old[j]
            self.b[0]=self.T_hot
            self.T_new[:]=scipy.linalg.solve(self.A, self.b)[:]
            self.T_old[:]=self.T_new[:]       
            

    def Graphy(self):
        plt.title("Temperature of the bar at different times", fontsize=18)
        plt.xlabel('X Coordinate (cm)', fontsize=13)
        plt.ylabel('Heat, K', fontsize=13)
        plt.show()
    
    
    

D=Rod_2b(5, 30, 50)
D.Jacobi_Euler()
D.Graphy()
print("Every different line is the rod at a different time stamp. The darker the shade of blue the further evolved")
print("the rod is.")


# This code is ultimately the same as above. However the main differences are the matrix of coefficients, A, and I took out the line of code which resets the final row of the b vector to be 273K again in the iteration.
# 
# On the last row of the matrix A, instead of there being a 1 in the last column as before in 2a there is a -2F, +2F+1, in the penultimate and final column respectively. 
# The reason I did this is because with the rod no longer in and ice bath the whole length of the rod, past the bit in the 1273K furnace, should have the coefficients applied to it. The problem with this is that the final row in the matrix A doesn't have columns for all 3 of the necessary coefficients. In total the coefficients are -F, +2F+1, -F, but since there isn't a column for the final -F, I decided to add it to the first -F. This seemed to give results as expected, i.e. the whole length of the rod slowly heating up to ~1273K.
# 
# The line of code which resets the last row of the vector column b to 273K for every new time step was of course necessary in 2a, but not helpful for 2b, where every row of the b column vector needs to be allowed to iterate freely.
# 
# The results shown above are as mentioned above, as expected. The whole length of the rod slowly heats up to 1273K as opposed to a uniform distriubution from 1273-->273K. However I once again have not put in the fourier series to act as an analytical solution to compare it too.

# ## Extensions
# 
# There are many possible extensions to this assignment, for example : MUST BE DONE BELOW
# * Model the field in more complex arrangements than the parallel plate capacitor in 1b). Tick
# * Model a point charge using the code from Q1? What are the problems/challenges in doing so? Second Q
# * Demonstrate that the explicit method in Q2 is unstable for some choices of $\delta t$ and $h$.
# * Implement higher-order methods (eg. Crank-Nicolson, which includes a 2nd order difference for the spatial derivative).
# 
# You are advised to discuss any extensions with your demonstrator or the unit director.  If you wish to include any extensions, please do so *below* this cell.

# In[7]:


print("I believe the code below (same as in 1b) answers the first and second bullet point.")
print("To see my written answers on these points, see the text box to 1b please.")

Input='0'
while Input!='Quit':
    Input=input('Enter a choice of, "Parallel Plate", "Singularity", "Disc" or "Quit", to quit: ')
    if Input=="Parallel Plate":
        print('You have entered: ', '"',Input,'"')
        Input=Parallel_Plate(0.1, 10, 5)
        Input1='0'
        Input1=input("Type a value of the distance separating the two plates as a fraction of the width of the box encapsulating them, eg. 0.2:  ")
        Input2='0'
        Input2=input("Type a value for the length of the plates as a fraction of the length of the box which encapsulates them, eg. 0.9:   ")
        print("-- This may take a while --")
        Input.Graph('Parallel Plate', Input.Jacob(Input.Bound(float(Input2), float(Input1)), 1e-3), 'seismic')
        
    elif Input=="Singularity":
        print('You have entered: ', '"',Input,'" -- ',  'This may take a while...')
        Input3='0'
        Input3=input("Give the x coordinate of the Singularity on an axis going from 0 to 1:  ")
        Input4='0'
        Input4=input("Give the y coordinate of the Singularity on an axis going from 0 to 1:  ")
        Input=Singularity(0.1, 3, 5)
        Input.Graph('Singularity', Input.Jacob(Input.Bound(float(Input3), float(Input4)), 1e-3), 'afmhot')
            
    elif Input=="Disc":
        print('You have entered: ', '"',Input,'"')
        Input8='0'
        Input8=input("Give the resolution, the higher the resolution the longer this will take. In order to see how you can use a Disc to   model a singularity, I suggest resolution 200 with a radius of 0.01:   ")
        Input=Disc(1, float(Input8), 5)
        Input7='0'
        Input7=input("Type a value of radius of Disc as a fraction of the width of the box encapsulating them, eg. 0.2:  ")
        print("-- This may take a while --")
        Input.Graph('Disc', Input.Jacob(Input.Bound(0.5, 0.5, float(Input7)), 1e-3), 'afmhot')
        

