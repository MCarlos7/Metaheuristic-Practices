import numpy as np
import matplotlib.pyplot as plt

x = np.random.rand(30)
y = np.random.rand(30)

plt.scatter(x, y)
plt.grid(True)
plt.title('Puntos Aleatorios')
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.show()
