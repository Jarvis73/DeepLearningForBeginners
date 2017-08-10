import numpy as np
INF = 2**31-1

image = np.arange(4).reshape((1, 2, 2))
image2 = np.arange(4).reshape((1, 2, 2)) * 2
gridX = np.linspace(0, 3, num=3, endpoint=False)
gridY = np.linspace(0, 4, num=4, endpoint=False)
vy, vx = np.meshgrid(gridX, gridY)
vx = vx.reshape((1, 4, 3))
vy = vy.reshape((1, 4, 3))

print(np.vstack((vx, vy)))
