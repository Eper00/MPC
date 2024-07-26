import numpy as np
import matplotlib.pyplot as plt
from one_set_simulation import realsystem
import random as rn
from scipy.spatial import ConvexHull

terminal_sets=np.empty((0, 6))
population=9800000
def inspect(test_set):
    for i in range(len(test_set)-1):
        if (test_set[i]<=test_set[i+1]):
            return 1
    return 0
y0=np.zeros(6)
for mapping in range(10):
    sum=0
    for i in range (6):
        y0[i]=rn.random()
        sum=sum+y0[i]
    y0=y0/sum
    sol=realsystem(y0)
    H=sol.y[5]
    opinion=inspect(H)
    if (opinion==0):
        terminal_sets=np.vstack((terminal_sets,y0))

terminal_sets=terminal_sets
colors = terminal_sets[:, 3:6]

for i in range (len (colors)):
    sum=0
    for t in range (3):
        sum=sum+colors[i,t]
    colors[i,:]=colors[i,:]/sum
    
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(terminal_sets[:,0], terminal_sets[:,1], terminal_sets[:,2],c=colors)
plt.show()
print(np.shape(terminal_sets))
hull = ConvexHull(terminal_sets)
# A burkoló vertexei
print("Vertices of the convex hull:")
print(hull.vertices)


