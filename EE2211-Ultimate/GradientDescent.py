def GradientDescent(f, f_prime, initial, learning_rate, num_iters, round_output=True, decimals=4):
    import numpy as np

    x0 = np.asarray(initial, dtype=float)
    steps = np.array([x0])

    for iteration in range(num_iters):
        grad = np.asarray(f_prime(steps[iteration]), dtype=float)
        new_step = steps[iteration] - learning_rate * grad
        steps = np.vstack((steps, new_step))

    fvals = np.array([f(s) for s in steps], dtype=float)
    grads = np.array([f_prime(s) for s in steps], dtype=float)

    if round_output:
        steps = np.round(steps, decimals)
        fvals = np.round(fvals, decimals)
        grads = np.round(grads, decimals)

    return steps, fvals, grads