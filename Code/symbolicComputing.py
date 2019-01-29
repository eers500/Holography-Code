# -*- coding: utf-8 -*-
from sympy import *
x, y, z = symbols('x y z')
f = x+y+z
a = float(f.subs([(x,1), (y,1), (z,1)]))
#%%
from sympy import *
import numpy as np
x, y = symbols('x y')
f = x+y
xx = np.arange(0,1,0.1)
yy = xx

a = np.array(f.subs([(x,xx),(y,yy)]))
