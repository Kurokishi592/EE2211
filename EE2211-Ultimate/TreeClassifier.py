def TreeClassifier(X_train, X_test, y_train, y_test, criterion, max_depth):
    from sklearn import tree
    from sklearn.metrics import accuracy_score
    import matplotlib.pyplot as plt
    # can be any number of features
    # X_train is of shape (n_samples, n_features)
    # X_test is of shape (n_samples, n_features)
    
    # encoded in 0, 1, 2, ...
    # y_train is of shape (n_samples,)
    # y_test is of shape (n_samples,)
    
    # fit tree
    dtree = tree.DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    dtree = dtree.fit(X_train, y_train) 
    
    # predict
    y_trainpred = dtree.predict(X_train)
    y_testpred = dtree.predict(X_test)
    
    # print accuracies
    print("[treeClassifier] Training accuracy: ", accuracy_score(y_train, y_trainpred))
    print("[treeClassifier] Test accuracy: ", accuracy_score(y_test, y_testpred))    

    # Plot tree
    tree.plot_tree(dtree)
    plt.show()
    
    # tree decision is given with respect to X[feature index]
    # a plot of the tree with [class 0, class 1, class 2, ...] as the amount in each class