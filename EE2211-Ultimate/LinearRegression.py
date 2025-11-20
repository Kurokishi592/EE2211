def linear_regression(X, y, X_test):
    import numpy as np
    if X.shape[1]<X.shape[0]:
        system="overdetermined"
    elif X.shape[1]>X.shape[0]:
        system="underdetermined"
    else:
        system="full rank"
    print(system, "system \n")
    np.set_printoptions(precision=4, suppress=True)
    if system=="overdetermined":
        w=np.linalg.inv(X.T@X)@X.T@y
    elif system=="underdetermined":
        w=X.T@np.linalg.inv(X@X.T)@y
    else:
        w=np.linalg.inv(X)@y
    print("w is (first row is for bias): \n", w, "\n")

    y_calculated=X@w
    y_difference_square=np.square(y_calculated-y)
    sum_of_square=sum(y_difference_square)
    mean_squared_error=sum_of_square/y.shape[0]
    print("square error is", sum_of_square)
    print("MEAN square error is", mean_squared_error, "\n")

    y_predicted_train=X@w
    print("y_train_predicted is\n" , np.round(y_predicted_train, 4), "\n")
    print("if binary classification, y_train_predicted_classified is\n" , np.sign(y_predicted_train), "\n")

    y_predicted_test=X_test@w
    print("y_test_predicted is\n", np.round(y_predicted_test, 4), "\n")
    print("if binary classification, y_test_predicted_classified is\n", np.sign(y_predicted_test), "\n")

    # print("X rank:", np.linalg.matrix_rank(X))
    # result=np.hstack((X,y))
    # print("X|y rank: ", np.linalg.matrix_rank(result))


    return(system, w, sum_of_square, mean_squared_error, y_predicted_train, y_predicted_test)
