import numpy as np
from scipy.interpolate import interp1d

# Datos de ejemplo
x = np.array([0, 1, 2, 3, 4, 5])
y = np.array([1, 3, 2, 4, 3, 7])

# Crear la función de interpolación
f = interp1d(x, y, kind='cubic')  # Usa interpolación cúbica

# Ajustar un polinomio de grado 3 (cúbico)
coef = np.polyfit(x, y, 3)
poly_eq = np.poly1d(coef)

# Mostrar la ecuación
print("Ecuación polinómica ajustada:")
print(poly_eq)


# Evaluar la función en nuevos puntos
x_nuevo = np.linspace(0, 5, 100)
y_nuevo = f(x_nuevo)

# Visualizar los resultados (opcional)
import matplotlib.pyplot as plt

plt.plot(x, y, 'o', label='Puntos originales')
plt.plot(x_nuevo, y_nuevo, label='Función interpolada')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Interpolación cúbica')
plt.legend()
plt.grid(True)

plt.show()