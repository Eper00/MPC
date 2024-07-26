import numpy as np
from scipy.spatial import ConvexHull

# Definiáljuk a ponthalmazt a 6D térben
points = np.random.rand(30, 6)  # 30 pont a 6 dimenziós térben

# Számoljuk ki a konvex burkolót
hull = ConvexHull(points)

# A burkoló vertexei
print("Vertices of the convex hull:")
print(hull.vertices)

# Nyomtatás a síkegyenletekről (hyperplanes)
print("\nEquations of the hyperplanes:")
print(hull.equations)

# Pontok az eredeti halmazból, amelyek a burkoló határát képezik
print("\nPoints that form the convex hull:")
print(points[hull.vertices])
