import numpy as np
import matplotlib.pyplot as plt
from one_set_simulation import realsystem
import random as rn
from scipy.spatial import ConvexHull, distance

# Initialize terminal_sets array
terminal_sets = np.empty((0, 5))

# Population size (not used directly in this code)
population = 9800000

def inspect(test_set):
    for i in range(len(test_set) - 1):
        if test_set[i] <= test_set[i + 1]:
            return 1
    return 0

# Initial conditions
y0 = np.zeros(6)
res = np.zeros(5)

# Run the simulation
for mapping in range(1000):
    sum = 0
    for i in range(6):
        y0[i] = rn.random()
        sum += y0[i]
    y0 = y0 / sum
    sol = realsystem(y0)
    H = sol.y[5]
    opinion = inspect(H)
    if opinion == 0:
        res = np.delete(y0, 4)
        terminal_sets = np.vstack((terminal_sets, res))

# Create ConvexHull object
hull = ConvexHull(terminal_sets[:, 0:5])

x0 = np.array([1-10/population,10/population,0,0,0])
distances = distance.cdist([x0], terminal_sets[hull.vertices, :5], 'euclidean')

min_distance = np.min(distances)

print("Minimum distance from the point to the convex hull:", min_distance)
