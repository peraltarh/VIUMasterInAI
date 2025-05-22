import math
import time


def fibonacci(n):
    """Calculate the Fibonacci number at position n (0-indexed)."""
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
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
    """Calculate the Fibonacci number at position n using Binet's Formula."""
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
    sqrt5 = math.sqrt(5)
    phi = (1 + sqrt5) / 2
    psi = (1 - sqrt5) / 2
    return round(pow(phi, n) - pow(psi, n) / sqrt5)
    

start_time = time.time()
print (fibonacci(1000)) 
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
print (fibonacci_binet(1000))
print("--- %s seconds ---" % (time.time() - start_time))


# Example usage:
# print(fibonacci(10))  # Output: 55