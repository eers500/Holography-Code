# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 12:27:15 2019

@author: erick
"""
#%%
import time
from time import sleep
import dask


def times2(x):
    sleep(1)
    return 2*x

data = [1, 2, 3, 4, 5, 6, 7 ,8 , 9, 10]
y = []

#%%
#time
T0 = time.time()
for i in data:
    x = times2(i)
    y.append(x)
total = sum(y)
print('total = ', total)    
print(time.time() - T0)

#%%
T0 = time.time()
for i in data:
    x = dask.delayed(times2)(i)
    y.append(x)
    
res = dask.delayed(sum)(y)
total = res.compute()
print('total = ', total)    
print(time.time() - T0)