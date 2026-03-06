import numpy as np
import matplotlib as plt

#Parametros
num_puntos = 60
numDim = 2

#Vector de entertos centrado al rededro de 0 
rangoIndices = np.arange(-(num_puntos-1)/2, (num_puntos-1)/2 + 1)

#Matriz inicial vacia 
matrizPerm = np.zeros((num_puntos, numDim))

#Generar permutacion aleatoria de los indices
for dim in range(numDim):
    matrizPerm[:, dim] = np.random.permutation(rangoIndices)

#Generar numeros alatorios uniforme para los ajustes dentro de las celulas
ajusteRand = np.random.uniform(num_puntos, numDim)

#Normalizar los valores a los intervalos [0;1]
puntosNorm = (matrizPerm + (num_puntos - 1) / 2 + ajusteRand) / num_puntos

#Graficar los puntos en 2d 
plt.plot(puntosNorm[:, 0], puntosNorm[:, 1], 'o', color='blue', markersize=8, label='Puntos LHS')

#Agregar marcas en los ejes X y Y
plt.xticks(np.arange(np.arange(0, 1 +1/num_puntos, 1/num_puntos)))
plt.yticks(np.arange(np.arange(0, 1 +1/num_puntos, 1/num_puntos)))

#Escalar los puntos al rango [-3,3]
puntosEsc = np.zeros_like(puntosNorm)
puntosEsc[:, 0] = puntosNorm[:, 0] * 6 -3
puntosEsc[:, 1] = puntosNorm[:, 1] * 6 -3

#Mostar la cuadrícula en la grafica 
plt.grid(True)
plt.show()
 
#Matriz normalizada sin el ajuste aleatorio 
puntosNormSinRand = (matrizPerm + (num_puntos - 1) / 2) / num_puntos

#Matriz de transformaciones (ejemplo dado)
matrizTranz = np.array([[3, 1/2], [2, 1/2]])
