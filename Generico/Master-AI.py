import math
import time


def fibonacci(n):
    if n < 0:
        raise ValueError("N tiene que ser positivo.")
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for contador in range(2, n + 1):
            a, b = b, a + b
        return b

def fibonacci_binet(n):
    """Calculate the Fibonacci number at position n using Binet's Formula. Fn = ((1 + √5) / 2)^n - ((1 - √5) / 2)^n / √5 """
    if n < 0:
        raise ValueError("N tiene que ser positivo.")
    sqrt5 = math.sqrt(5)
    phi = (1 + sqrt5) / 2
    psi = (1 - sqrt5) / 2
    result = (phi**n - psi**n) / sqrt5
    return round(result)
    


# Example : 1 1 2 3 5 8 13 21 34 55
# print(fibonacci(10))  # Output: 55
print("Inicio")
print(fibonacci(10))
print(fibonacci_binet(10))
print("Fin")