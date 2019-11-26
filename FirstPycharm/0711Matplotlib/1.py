import numpy as np
from matplotlib import pyplot as plt

x = np.linspace(-np.pi, np.pi, 100)
c = np.cos(x)
s = np.sin(x)

plt.plot(x,c, 'r*')
plt.plot(x,s, 'bo')

plt.show()