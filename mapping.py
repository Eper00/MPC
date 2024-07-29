import numpy as np
from one_set_simulation import realsystem
import random as rn
def inspect(test_set):
    for i in range(len(test_set) - 1):
        if test_set[i] <= test_set[i + 1]:
            return 1
    return 0
def kth_largest(a, k): 
    k = len(a) - 1 - k 
    return np.partition(a, k)[k] 
def map (number_of_points):
    y0 = np.zeros(6)
    terminal_sets = np.empty((0, 6))
    for mapping in range(number_of_points):
        sum = 0
        for i in range(6):
            if i==0:
                y0[i] = rn.random()
            sum += y0[i]
        y0 = y0 / sum
        sol = realsystem(y0)
        H = sol.y[5]
        opinion = inspect(H)
        if opinion == 0:
            terminal_sets = np.vstack((terminal_sets, y0))
            
            
    th=0
        
    while th<len(terminal_sets):
        candidate=kth_largest(terminal_sets[:,0],th)
        element=np.where(terminal_sets == candidate)[0]    
        
        for iter in range (50):
            for i in range (6):
                if (i==0):
                    y0[i]=terminal_sets[element,0]
                else:
                    y0[i]=rn.uniform(0,0.17*y0[0])
            H=realsystem(y0).y[5]

            if (inspect(H)==1):
                break
        if (inspect(H)==0):
            return y0[0]
        
        th=th+1
            
    return 0
def get_terminal_state():
    res=0
    while (res==0):
        res=map(1000)
    return res