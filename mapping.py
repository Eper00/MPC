import numpy as np
from one_set_simulation import realsystem
import random as rn
def inspect(test_set):
    for i in range(len(test_set) - 1):
        if test_set[i] <= test_set[i + 1]:
            return 1
    return 0
def map (number_of_points):
    y0 = np.zeros(6)
    terminal_sets = np.empty((0, 6))
    for mapping in range(number_of_points):
        sum = 0
        for i in range(6):
            y0[i] = rn.random()
            sum += y0[i]
        y0 = y0 / sum
        sol = realsystem(y0)
        H = sol.y[5]
        opinion = inspect(H)
        if opinion == 0:
            terminal_sets = np.vstack((terminal_sets, y0))
    return terminal_sets

