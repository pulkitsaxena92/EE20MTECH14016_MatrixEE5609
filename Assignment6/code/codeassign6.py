import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
len = 100
y = np.linspace(-4,4,len)
V = np.array(([9,-12],[-12,16]))
u = np.array(([-9,-50.5]))
f = 19
O = np.array(([0,0]))
D_vec,P = LA.eig(V)
D = np.diag(D_vec)
p = P[:,0]
eta = 2*u@p
foc = eta/D_vec[1]
x = y**2/foc
cA = np.vstack((u+eta*0.5*p,V))
cb = np.vstack((-f,(eta*0.5*p-u).reshape(-1,1)))
c = LA.lstsq(cA,cb,rcond=None)[0]
c = c.flatten()
P=-P
c1 = np.array(([(u@V@u-2*D_vec[0]*u@u+D_vec[0]**2*f)/(eta*D_vec[0]**2),0]))
xStandardparab = np.vstack((x,y))
xActualparab = P@xStandardparab + c[:,np.newaxis]
parab_coords = np.vstack((O,c)).T
plt.scatter(parab_coords[0,:],parab_coords[1,:])
vert_labels = ['$O$','$c (-1.16,0.88)$']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, 
                 (parab_coords[0,i], parab_coords[1,i]),
                 textcoords="offset points",
                 xytext=(0,5), 
                 ha='center')
plt.plot(xActualparab[0,:],xActualparab[1,:],label='Parabola',color='r')
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid()
plt.axis('equal')
plt.show()


    
