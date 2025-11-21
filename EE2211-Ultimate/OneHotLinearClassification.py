def onehot_linearclassification(X, y, X_test):
    import numpy as np
    from sklearn.preprocessing import OneHotEncoder
    from evaluation_metrics import (
        compute_multiclass_metrics, print_multiclass_metrics,
    )
    onehot_encoder = OneHotEncoder(sparse_output=False)
    y_onehot = onehot_encoder.fit_transform(y)
    print("the onehot encoded y is:\n", y_onehot)
    print("")
    np.set_printoptions(precision=4, suppress=True)
    #linear regression process
    if X.shape[1]<X.shape[0]:
        system="overdetermined"
    elif X.shape[1]>X.shape[0]:
        system="underdetermined"
    else:
        system="full rank"
    print(system, "system \n")

    if system=="overdetermined":
        w=np.linalg.inv(X.T@X)@X.T@y_onehot
    elif system=="underdetermined":
        w=X.T@np.linalg.inv(X@X.T)@y_onehot
    else:
        w=np.linalg.inv(X)@y_onehot 
    print("w is (first row is for bias): \n", w, "\n")

    y_calculated=X@w
    print("y_train_raw is (col pos of largest per row is the class index):\n", y_calculated, "\n")
    y_train_pred = np.argmax(y_calculated,axis=1)
    print("y_train_classified (transpose urself for argmax of each row. Remember + 1 if qn's class starts with 1) is\n", y_train_pred, "\n")
    # Metrics
    y_true_cls = np.argmax(y_onehot, axis=1)
    print("[onehot_linearclassification] Train multiclass metrics:")
    print_multiclass_metrics(compute_multiclass_metrics(y_true_cls, y_train_pred))
    # y_difference_square=np.square(y_calculated-y)
    # sum_of_square=sum(y_difference_square)
    # mean_squared_error=sum_of_square/y.shape[0]
    # print("square error is", sum_of_square)
    # print("MEAN square error is", mean_squared_error, "\n")

    y_predicted=X_test@w
    y_predicted_classes=np.argmax(y_predicted,axis=1)
    print("y_test_raw is (col pos of largest per row is the class index):\n", y_predicted, "\n")
    print("y_test_classified (transpose urself for argmax of each row. Remember + 1 if qn's class starts with 1) is\n", y_predicted_classes, "\n")


