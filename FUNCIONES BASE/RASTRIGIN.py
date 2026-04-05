import numpy as np 
import matplotlib.pyplot as plt 

#RANGOS
x = np.linspace(-5.12, 5.12, 200)
y = np.linspace(-5.12, 5.12, 200)
X, Y = np.meshgrid(x, y)

#FUNCIONES RASTRIGIN
A = 10
Z = A * 2 + (X**2 - A * np.cos(2 * np.pi * X)) + (Y**2 - A * np.cos(2 * np.pi * Y))

#GRAFICA 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='jet', edgecolor='none')

#ETIQUETAS
ax.set_title('Función Rastrigin:', fontsize=16)
ax.set_xlabel('Eje X', fontsize=12) 
ax.set_ylabel('Eje Y', fontsize=12)
ax.set_zlabel('f(x,y)', fontsize=12)
fig.colorbar(surf, shrink=0.5, aspect=5)

#VISTA
ax.view_init(elev=30, azim=45)
plt.show()

#CURVAS DE NIVEL
plt.figure(figsize=(7, 6))
contour = plt.contourf(X, Y, Z, levels=50, cmap='jet')
plt.title('Curvas de Nivel de la Función Rastrigin', fontsize=16)
plt.xlabel('Eje X', fontsize=12)
plt.ylabel('Eje Y', fontsize=12)
plt.colorbar(contour)
plt.show()