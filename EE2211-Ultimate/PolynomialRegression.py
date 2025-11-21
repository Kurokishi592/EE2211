def polynomial_regression(X,y,order,X_test):
    import numpy as np
    from sklearn.preprocessing import PolynomialFeatures
    from evaluation_metrics import (
        compute_regression_metrics, print_regression_metrics,
        compute_binary_classification_metrics, print_binary_classification_metrics,
        compute_multiclass_metrics, print_multiclass_metrics,
    )
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
    # y_difference_square=np.square(P_train_predicted-y)
    # sum_of_square=sum(y_difference_square)
    # mean_squared_error=sum_of_square/y.shape[0]
    # print("square error is", sum_of_square)
    # print("MEAN square error is", mean_squared_error, "\n")
    
        # Metrics: regression always, plus classification when labels indicate it
    try:
        train_reg = compute_regression_metrics(y_true=y, y_pred=P_train_predicted)
        print("[polynomial_regression] Train regression metrics:")
        print_regression_metrics(train_reg)
    except Exception:
        pass

    # Classification metrics detection
    y_arr = np.asarray(y)
    if y_arr.ndim == 2 and y_arr.shape[1] > 1:
        # Multiclass one-hot
        y_true_cls = np.argmax(y_arr, axis=1)
        y_pred_cls = np.argmax(P_train_predicted, axis=1)
        print("\n[polynomial_regression] Train multiclass metrics:")
        print_multiclass_metrics(compute_multiclass_metrics(y_true_cls, y_pred_cls))
    else:
        vals = np.unique(y_arr.ravel())
        if set(vals).issubset({-1, 1}):
            y_pred_cls = np.sign(P_train_predicted).ravel()
            pos = 1
            print("\n[polynomial_regression] Train binary metrics:")
            print_binary_classification_metrics(
                compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=pos)
            )
        elif set(vals).issubset({0, 1}):
            y_pred_cls = (P_train_predicted.ravel() >= 0.5).astype(int)
            pos = 1
            print("\n[polynomial_regression] Train binary metrics:")
            print_binary_classification_metrics(
                compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=pos)
            )
    print("")

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
    
    # Test metrics
    # try:
    #     test_reg = compute_regression_metrics(y_true=y, y_pred=y_predicted)  # train metrics already shown
    # except Exception:
    #     pass

    # if y_arr.ndim == 2 and y_arr.shape[1] > 1:
    #     y_true_cls = np.argmax(y_arr, axis=1)
    #     y_pred_cls = np.argmax(y_predicted, axis=1)
    #     print("[ridge_regression] Test multiclass metrics:")
    #     print_multiclass_metrics(compute_multiclass_metrics(y_true_cls, y_pred_cls))
    # else:
    #     vals = np.unique(y_arr.ravel())
    #     if set(vals).issubset({-1, 1}):
    #         y_pred_cls = np.sign(y_predicted).ravel()
    #         print("[ridge_regression] Test binary metrics:")
    #         print_binary_classification_metrics(
    #             compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=1)
    #         )
    #     elif set(vals).issubset({0, 1}):
    #         y_pred_cls = (y_predicted.ravel() >= 0.5).astype(int)
    #         print("[ridge_regression] Test binary metrics:")
    #         print_binary_classification_metrics(
    #             compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=1)
    #         )


    # if single class classification
    # y_classified = np.sign(y_predicted)
    # print("y_classified is", y_classified)
    #
    # return(system, P, w, y_predicted, y_classified)

    # print("P rank:", np.linalg.matrix_rank(P))
    # result=np.hstack((P,y))
    # print("P|y rank: ", np.linalg.matrix_rank(result))
