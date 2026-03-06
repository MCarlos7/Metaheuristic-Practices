import numpy as np 
import matplotlib.pyplot as plt 

#RANGOS
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)

#FUNCIONES sphere
Z = X**2 + Y**2

#GRAFICA 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none')

#ETIQUETAS
ax.set_title('Función Sphere: $f(x, y) = x^2 + y^2$', fontsize=16)
ax.set_xlabel('Eje X', fontsize=12) 
ax.set_ylabel('Eje Y', fontsize=12)
ax.set_zlabel('Eje Z', fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=5)

#VISTA
ax.view_init(elev=30, azim=45)
plt.show()