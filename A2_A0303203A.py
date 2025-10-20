import numpy as np

# Please replace "MatricNumber" with your actual matric number here and in the filename
def A2_A0303203A(N):
    """
    Input type
    :N type: int

    Return type
    :X_train type: numpy.ndarray of size (number_of_training_samples, 4)
    :y_train type: numpy.ndarray of size (number_of_training_samples,)
    :X_test type: numpy.ndarray of size (number_of_test_samples, 4)
    :y_test type: numpy.ndarray of size (number_of_test_samples,)
    :Ytr type: numpy.ndarray of size (number_of_training_samples, 3)
    :Yts type: numpy.ndarray of size (number_of_test_samples, 3)
    :Ptrain_list type: List[numpy.ndarray]
    :Ptest_list type: List[numpy.ndarray]
    :w_list type: List[numpy.ndarray]
    :error_train_array type: numpy.ndarray
    :error_test_array type: numpy.ndarray
    """

    # your code goes here
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.6, random_state=N)
    
    from sklearn.preprocessing import OneHotEncoder
    onehot_encoder = OneHotEncoder(sparse_output=False)
    reshaped = y_train.reshape(len(y_train), 1)
    Ytr = onehot_encoder.fit_transform(reshaped)
    reshaped = y_test.reshape(len(y_test), 1)
    Yts = onehot_encoder.fit_transform(reshaped)
    
    from sklearn.preprocessing import PolynomialFeatures
    reg_factor = 0.0001
    Ptrain_list = []
    Ptest_list = []
    w_list = []
    error_train_array = np.zeros(10)
    error_test_array = np.zeros(10)
    
    for order in range (1, 11):
        poly = PolynomialFeatures(order)
        P = poly.fit_transform(X_train)
        Ptest = poly.fit_transform(X_test) 
        # dual if rows <= cols, primal otherwise
        if P.shape[0] <= P.shape[1]:
            w = P.T @ np.linalg.inv(P @ P.T + reg_factor * np.eye(P.shape[0])) @ Ytr
        else:
            w = np.linalg.inv(P.T @ P + reg_factor * np.eye(P.shape[1])) @ P.T @ Ytr
        Ptrain_list.append(P)
        Ptest_list.append(Ptest)
        w_list.append(w)
        # error keeps track of wrong classification count of prediction
        error_train_array[order-1] = np.sum(np.argmax(P @ w, axis=1) != y_train)
        error_test_array[order-1] = np.sum(np.argmax(Ptest @ w, axis=1) != y_test)

    # return in this order
    return X_train, y_train, X_test, y_test, Ytr, Yts, Ptrain_list, Ptest_list, w_list, error_train_array, error_test_array
