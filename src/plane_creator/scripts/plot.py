import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

soa = np.array([[centroid[0], centroid[1], centroid[2],  xVector[0] ,  xVector[1],  xVector[2]], [centroid[0], centroid[1], centroid[2],  yVector[0] ,  yVector[1],  yVector[2]], [centroid[0], centroid[1], centroid[2],  zVector[0] ,  zVector[1],  zVector[2]], [0,0,0,0,0,0.1]])

X, Y, Z, U, V, W = zip(*soa)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W)
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
plt.savefig('plot.png')
