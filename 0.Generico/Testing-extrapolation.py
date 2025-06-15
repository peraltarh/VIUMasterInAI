import numpy as np
import matplotlib.pyplot as plt

# Puntos de interpolación (Valores dados para la ejecución del programa)
x = [0, 1, 2]
y = [1, 3, 0]

# Función para calcular el polinomio de Lagrange
def lagrange(x, y, xi):
    n = len(x)
    result = 0.0
    for i in range(n):
        term = y[i]
        for j in range(n):
            if i != j:
                term *= (xi - x[j]) / (x[i] - x[j])
        result += term
    return result

# Generar puntos para graficar el polinomio
xi = np.linspace(0, 2, 100)  # Valores de x para la gráfica
yi = [lagrange(x, y, xi_val) for xi_val in xi]

# Graficar los puntos de interpolación
plt.scatter(x, y, color='red', label='Puntos de interpolación')

# Graficar el polinomio interpolante
plt.plot(xi, yi, label='Polinomio de Lagrange')

# Configuración del gráfico
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)

# Esta es la grafica del polinomio:
plt.show()