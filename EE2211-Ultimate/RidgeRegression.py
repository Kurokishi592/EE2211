def ridge_regression(X, y, LAMBDA, X_test, form="auto"):
    import numpy as np
    np.set_printoptions(precision=4, suppress=True)
    if form=="auto":
        if X.shape[1] < X.shape[0]:
            system = "overdetermined"
            form = "primal form"
        elif X.shape[1] > X.shape[0]:
            system = "underdetermined"
            form = "dual form"
        else:
            system = "full rank"
    else:
        if X.shape[1] < X.shape[0]:
            system = "overdetermined"
        elif X.shape[1] > X.shape[0]:
            system = "underdetermined"
        else:
            system = "full rank"

    print(system, "system   ", form)
    print("")

    if form=="primal form":
        I = np.identity(X.shape[1])
        w = np.linalg.inv(X.T @ X+LAMBDA*I) @ X.T @ y
    elif form == "dual form":
        I = np.identity(X.shape[0])
        w = X.T @ np.linalg.inv(X @ X.T+LAMBDA*I) @ y
    else:
        w = np.linalg.inv(X) @ y

    print("w (first row is for bias): ")
    print(w)
    print("")

    y_calculated=X@w
    print("y_train_predicted is: \n", np.round(y_calculated, 4), "\n")
    print("if one hot encoding multi-class classification, y_train_classes are (transpose urself for argmax of each row): \n", np.argmax(y_calculated, axis=1), "\n")
    print("if binary classification, y_train_predicted_classified is\n" , np.sign(y_calculated), "\n")
    y_difference_square=np.square(y_calculated-y)
    sum_of_square=sum(y_difference_square)
    mean_squared_error=sum_of_square/y.shape[0]
    print("square error is", sum_of_square)
    print("MEAN square error is", mean_squared_error, "\n")

    y_predicted=X_test@w
    print("y_test_predicted is\n", np.round(y_predicted, 4), "\n")
    print("if one hot encoding multi-class classification, y_test_classes are (transpose urself for argmax of each row): \n", np.argmax(y_predicted, axis=1), "\n")
    print("if binary classification, y_test_predicted_classified is\n", np.sign(y_predicted), "\n")
