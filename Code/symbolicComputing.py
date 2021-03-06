#%% Symbolic computing and 3D plot (surface)
import numpy as np
from sympy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

x, y, z = symbols('x y z')
x1 = np.arange(-3,3,1)
y1 = x1

xx, yy = np.meshgrid(x1,y1)

expr = z*exp(-(x**2+y**2))

f = lambdify([x,y,z], expr,'numpy')
c = f(xx, yy,z)

#fig = plt.figure()
#ax = fig.gca(projection = '3d')
#ax.plot_surface(xx, yy, c, cmap = 'jet')

#%% As MATLABs function handles
import numpy as np
fn = lambda x,y,z: np.exp(-(x**2+y**2))
a = np.arange(-1,1,0.1)
xx,yy,zz = np.meshgrid(a,a,a)
z = fn(xx,yy,zz)