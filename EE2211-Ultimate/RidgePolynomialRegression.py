def ridge_poly_regression(X,y,LAMBDA,order, form, X_test):
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    np.set_printoptions(precision=4, suppress=True)
    poly = PolynomialFeatures(order) # auto includes bias term
    P = poly.fit_transform(X)
    print("the number of parameters: ", P.shape[1])
    print("the number of samples: ", P.shape[0])
    if form=="auto":
        if P.shape[1] < P.shape[0]:
            system = "overdetermined"
            form = "primal form"
        elif P.shape[1] > P.shape[0]:
            system = "underdetermined"
            form = "dual form"
        else:
            system = "full rank"
    else:
        if P.shape[1] < P.shape[0]:
            system = "overdetermined"
        elif P.shape[1] > P.shape[0]:
            system = "underdetermined"
        else:
            system = "full rank"

    print(system, "system   ", form)
    print("")
    print("the polynomial transformed matrix P is:")
    print(P)
    print("")

    if form=="primal form":
        I = np.identity(P.shape[1])
        w = np.linalg.inv(P.T @ P+LAMBDA*I) @ P.T @ y
    elif form == "dual form":
        I = np.identity(X.shape[0])
        w = P.T @ np.linalg.inv(P @ P.T+LAMBDA*I) @ y
    else:
        w = np.linalg.inv(P) @ y

    print("w is (first row is order 0): ")
    print(w)
    print("")
    
    P_train_predicted=P@w
    print("y_train_predicted is: \n", np.round(P_train_predicted, 4), "\n")
    print("if one hot encoding multi-class classification, y_train_classes are (transpose urself for argmax of each row): \n", np.argmax(P_train_predicted, axis=1), "\n")
    print("if binary classification, y_train_predicted_classified is: \n", np.sign(P_train_predicted) , "\n")
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
    print("if binary classification, y_test_predicted_classified is: \n", np.sign(y_predicted), "\n")


    # if single class classification
    # y_classified = np.sign(y_predicted)
    # print("y_classified is", y_classified)
    #HI
    # return(system, P, w, y_predicted, y_classified))










def ridge_poly_regression_simplified(X,y,LAMBDA,order, form, X_test, y_test):
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(order)
    P = poly.fit_transform(X)
    # print("the number of parameters: ", P.shape[1])
    # print("the number of samples: ", P.shape[0])
    if form=="auto":
        if P.shape[1] < P.shape[0]:
            system = "overdetermined"
            form = "primal form"
        elif P.shape[1] > P.shape[0]:
            system = "underdetermined"
            form = "dual form"
        else:
            system = "full rank"
    else:
        if P.shape[1] < P.shape[0]:
            system = "overdetermined"
        elif P.shape[1] > P.shape[0]:
            system = "underdetermined"
        else:
            system = "full rank"

    # print(system, "system   ", form)
    # print("")
    # print("the polynomial transformed matrix P is:")
    # print(P)
    # print("")

    if form=="primal form":
        I = np.identity(P.shape[1])
        w = np.linalg.inv(P.T @ P+LAMBDA*I) @ P.T @ y
    elif form == "dual form":
        I = np.identity(X.shape[0])
        w = P.T @ np.linalg.inv(P @ P.T+LAMBDA*I) @ y
    else:
        w = np.linalg.inv(P) @ y

    # print("w is: ")
    # print(w)
    # print("")

    P_test = poly.fit_transform(X_test)
    # print("transformed test sample P_test is")
    # print(P_test)
    # print("")
    y_predicted = P_test @ w
    # print("y_predicted is")
    # print(y_predicted)

    P_train_predicted=P@w
    # print("y_train_predicted is: ", np.sign(P_train_predicted))
    y_difference_square=np.square(P_train_predicted-y)
    sum_of_square=sum(y_difference_square)
    mean_squared_error=sum_of_square/y.shape[0]
    # print("square error is", sum_of_square)
    print("ridge train MEAN square error is", mean_squared_error)

    P_test_predicted=P_test@w
    # print("y_train_predicted is: ", np.sign(P_train_predicted))
    y_difference_square=np.square(P_test_predicted-y_test)
    sum_of_square=sum(y_difference_square)
    mean_squared_error=sum_of_square/y_test.shape[0]
    # print("square error is", sum_of_square)
    print("ridge test MEAN square error is", mean_squared_error, "\n")

