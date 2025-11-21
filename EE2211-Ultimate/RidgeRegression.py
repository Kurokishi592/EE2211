def ridge_regression(X, y, LAMBDA, X_test, form="auto"):
    import numpy as np
    from evaluation_metrics import (
        compute_regression_metrics, print_regression_metrics,
        compute_binary_classification_metrics, print_binary_classification_metrics,
        compute_multiclass_metrics, print_multiclass_metrics,
    )
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
    # y_difference_square=np.square(y_calculated-y)
    # sum_of_square=sum(y_difference_square)
    # mean_squared_error=sum_of_square/y.shape[0]
    # print("square error is", sum_of_square)
    # print("MEAN square error is", mean_squared_error, "\n")
        # Metrics: regression always, plus classification when labels indicate it
    try:
        train_reg = compute_regression_metrics(y_true=y, y_pred=y_calculated)
        print("[ridge_regression] Train regression metrics:")
        print_regression_metrics(train_reg)
    except Exception:
        pass

    # Classification metrics detection
    y_arr = np.asarray(y)
    if y_arr.ndim == 2 and y_arr.shape[1] > 1:
        # Multiclass one-hot
        y_true_cls = np.argmax(y_arr, axis=1)
        y_pred_cls = np.argmax(y_calculated, axis=1)
        print("\n[ridge_regression] Train multiclass metrics:")
        print_multiclass_metrics(compute_multiclass_metrics(y_true_cls, y_pred_cls))
    else:
        vals = np.unique(y_arr.ravel())
        if set(vals).issubset({-1, 1}):
            y_pred_cls = np.sign(y_calculated).ravel()
            pos = 1
            print("\n[ridge_regression] Train binary metrics:")
            print_binary_classification_metrics(
                compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=pos)
            )
        elif set(vals).issubset({0, 1}):
            y_pred_cls = (y_calculated.ravel() >= 0.5).astype(int)
            pos = 1
            print("\n[ridge_regression] Train binary metrics:")
            print_binary_classification_metrics(
                compute_binary_classification_metrics(y_arr.ravel(), y_pred_cls, positive_label=pos)
            )
    print("")

    y_predicted=X_test@w
    print("y_test_predicted is\n", np.round(y_predicted, 4), "\n")
    print("if one hot encoding multi-class classification, y_test_classes are (transpose urself for argmax of each row): \n", np.argmax(y_predicted, axis=1), "\n")
    print("if binary classification, y_test_predicted_classified is\n", np.sign(y_predicted), "\n")
    
    # Test metrics
    # try:
    #     test_reg = compute_regression_metrics(y_true=y, y_pred=y_calculated)  # train metrics already shown
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
