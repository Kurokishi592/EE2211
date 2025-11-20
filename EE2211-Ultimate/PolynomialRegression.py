def polynomial_regression(X,y,order,X_test):
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    np.set_printoptions(precision=4, suppress=True)
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X) # auto includes bias term
    print("the number of parameters: ", P.shape[1])
    print("the number of samples: ", P.shape[0])
    if P.shape[1] < P.shape[0]:
        system = "overdetermined"
    elif P.shape[1] > P.shape[0]:
        system = "underdetermined"
    else:
        system = "full rank"
    print(system, "system")
    print("")
    print("the polynomial transformed matrix P is:")
    print(P)
    print("")

    if system == "overdetermined":
        w = np.linalg.inv(P.T @ P) @ P.T @ y
    elif system == "underdetermined":
        w = P.T @ np.linalg.inv(P @ P.T) @ y
    else:
        w = np.linalg.inv(P) @ y
    print("w is: ")
    print(w)
    print("")

    P_train_predicted=P@w
    print("y_train_predicted is:\n ", np.round(P_train_predicted, 4))
    print("")
    print("if one hot encoding multi-class classification, y_train_classes are (transpose urself for argmax of each row): \n", np.argmax(P_train_predicted, axis=1), "\n")
    print("if binary classification, y_train_predicted_classified is:\n ", np.sign(P_train_predicted), "\n")
    y_difference_square=np.square(P_train_predicted-y)
    sum_of_square=sum(y_difference_square)
    mean_squared_error=sum_of_square/y.shape[0]
    print("square error is", sum_of_square)
    print("MEAN square error is", mean_squared_error, "\n")

    P_test = poly.fit_transform(X_test)
    print("transformed test sample P_test is")
    print(P_test)
    print("")
    y_predicted = P_test @ w
    print("y_test_predicted is")
    print(np.round(y_predicted, 4))
    print("")
    print("if one hot encoding multi-class classification, y_test_classes are (transpose urself for argmax of each row): \n", np.argmax(y_predicted, axis=1), "\n")
    print("if binary classification, y_test_predicted_classified is:\n ", np.sign(y_predicted))


    # if single class classification
    # y_classified = np.sign(y_predicted)
    # print("y_classified is", y_classified)
    #
    # return(system, P, w, y_predicted, y_classified)

    # print("P rank:", np.linalg.matrix_rank(P))
    # result=np.hstack((P,y))
    # print("P|y rank: ", np.linalg.matrix_rank(result))
