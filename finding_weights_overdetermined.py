import numpy as np
import A1_A0303203A as grading

X = np.array([
    [5, -6],
    [2, 0],
    [4, 7],
    [11, -8]
])

y = np.array([
    [3],
    [-5.5],
    [9],
    [1]
])

InvXTX, w = grading.A1_A0303203A(X, y)

print("Inverse of X^T X:\n", InvXTX)
print("Least squares solution w:\n", w)
