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

f = lambdify([x,y], expr,'numpy')
c = f(xx, yy,)

fig = plt.figure()
ax = fig.gca(projection = '3d')
ax.plot_surface(xx, yy, c, cmap = 'jet')
