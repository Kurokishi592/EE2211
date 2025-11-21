def TreeRegressor(X_train, X_test, y_train, y_test, criterion, max_depth):
    '''
    Only from 1D to 1D regression
    '''
    import numpy as np
    from sklearn import tree
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt
    # X_train is of shape (n_samples,)
    # X_test is of shape (n_samples,)
    
    # encoded in 0, 1, 2, ...
    # y_train is of shape (n_samples,)
    # y_test is of shape (n_samples,)
    
    sort_index = X_train.argsort()
    X_train = X_train[sort_index]
    y_train = y_train[sort_index]
    
    sort_index = X_test.argsort()
    X_test = X_test[sort_index]
    y_test = y_test[sort_index]

    # scikit decision tree regressor
    scikit_tree = tree.DecisionTreeRegressor(criterion=criterion, max_depth=max_depth)
    scikit_tree.fit(X_train.reshape(-1,1), y_train) # reshape necessary because tree expects 2D array
    
    # predict
    y_trainpred = scikit_tree.predict(X_train.reshape(-1,1))
    y_testpred = scikit_tree.predict(X_test.reshape(-1,1))
    
    # print accuracies
    print("[treeRegressor] Training MSE: ", mean_squared_error(y_train, y_trainpred))
    print("[treeRegressor] Test MSE: ", mean_squared_error(y_test, y_testpred))
    
    # Plot
    plt.figure(0, figsize=[9,4.5])
    plt.rcParams.update({'font.size': 16})
    plt.scatter(X_train, y_train, c='steelblue', s=30)
    plt.plot(X_train, y_trainpred, color='red', lw=2, label='scikit-learn')
    plt.xlabel('X train')
    plt.ylabel('Y train and predict')
    plt.legend(loc='upper right',ncol=3, fontsize=10)
    plt.show()