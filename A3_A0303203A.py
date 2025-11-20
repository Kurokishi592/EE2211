import numpy as np


# Please replace "StudentMatriculationNumber" with your actual matric number here and in the filename
def A3_A0303203A(learning_rate, num_iters):
    """
    Input type
    :learning_rate type: float
    :num_iters type: int

    Return type
    :a_out type: numpy array of length num_iters
    :f1_out type: numpy array of length num_iters
    :b_out type: numpy array of length num_iters
    :f2_out type: numpy array of length num_iters
    :c_out type: numpy array of length num_iters
    :d_out type: numpy array of length num_iters
    :f3_out type: numpy array of length num_iters
    """
    # your code goes here
    a_out = np.zeros(num_iters)
    f1_out = np.zeros(num_iters)
    b_out = np.zeros(num_iters)
    f2_out = np.zeros(num_iters)
    c_out = np.zeros(num_iters)
    d_out = np.zeros(num_iters)
    f3_out = np.zeros(num_iters)
    
    def f1(a):
        return np.power(a, 5)
    
    a = 2.5
    
    for i in range(num_iters):
        grad = 5 * np.power(a, 4)
        a = a - learning_rate * grad
        a_out[i] = a
        f1_out[i] = f1(a)

    def f2(b):
        return (np.power(np.sin(b), 2))

    b = 0.5
    
    for i in range(num_iters):
        grad = 2 * np.sin(b) * np.cos(b)
        b = b - learning_rate * grad
        b_out[i] = b
        f2_out[i] = f2(b)

    def f3(c, d):
        return np.power(c, 3) + (np.power(d, 2) * np.sin(d))

    c = 2
    d = 4

    for i in range(num_iters):
        grad_c = 3 * np.power(c, 2)
        grad_d = 2 * d * np.sin(d) + np.power(d, 2) * np.cos(d)
        c = c - learning_rate * grad_c
        d = d - learning_rate * grad_d
        c_out[i] = c
        d_out[i] = d
        f3_out[i] = f3(c, d)

    # return in this order
    return a_out, f1_out, b_out, f2_out, c_out, d_out, f3_out 
