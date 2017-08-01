import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3, 3)
y = -x**5 + x**3 + 4*x
plt.plot(x, y)
plt.ylim(-10, 10)
plt.show()