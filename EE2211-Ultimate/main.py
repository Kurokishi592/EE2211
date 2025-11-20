from EnterMetrics import EnterMetrics
from LinearRegression import linear_regression
from PolynomialRegression import polynomial_regression
from RidgePolynomialRegression import ridge_poly_regression
from RidgeRegression import ridge_regression
from OneHotLinearClassification import onehot_linearclassification
from pearson_correlation import pearson_correlation
import numpy as np
from sklearn.metrics import mean_squared_error
from EnterMetrics import EnterMetrics

''' no need to add column of 1s to X for regression. X_fitted does it already. First row of w is w0.'''
X=np.array(
    [[0, 0],
     [1, 1],
     [1, 0],
     [0, 1]
   ]
);

''' 
for multi-class classification (poly, ridge, ridgepoly), manually key in the onehot encoding: class 0 = [1, 0, 0], class 1 = [0, 1, 0], class 2 = [0, 0, 1] ...
    only the function "onehot_linear" has auto one hot encoding, simply type the class number 0, 1, 2 ...etc 
    this code's argmax starts from class 0 not 1. 
for binary, all functions are the same, key in -1 or 1 will do.
'''
Y=np.array(
    [[-1],
     [-1],
     [1],
     [1]
     ]
);

''' same dont add one column of 1s to X_test for regression'''
X_test=np.array(
    [[0.1, 0.1],
     [0.9, 0.9],
     [0.1,0.9],
     [0.9,0.1]
    ]
)

''' only for linear, since for poly, the code will add the bias term internally (PolynomialFeatures from sklearn)'''
X_fitted=np.hstack((np.ones((len(X),1)),X))
X_test_fitted=np.hstack((np.ones((len(X_test),1)),X_test))

''' used for regression task, or binary classification task '''
# linear_regression(X_fitted,Y, X_test_fitted) 

''' used for multi-category classification task (auto one hot, key in 0, 1, 2 ... for y) '''
# onehot_linearclassification(X_fitted,Y,X_test_fitted) 

'''
used for regression tasks, binary classification and multi-category classification tasks (manually one hot encode y for multi-category)
    also, bias are auto in this order for poly fit by sklearn
        (for poly order 3) Bias (1) x1 x2 x1^2 x1*x2 x2^2     x1^3 x1^2*x2 x1*x2^2 x2^3
    but in lecture/tutorial, it may be
        (for poly order 2) Bias (1) x1 x2 x1*x2 x1^2 x2^2
    so read qn carefuly may need to reorder rows of w if they ask for it (and watch P's columns) accordingly
'''
# ridge_regression(X_fitted,Y,LAMBDA=0.1, X_test=X_test_fitted, form='auto') #linear model
# polynomial_regression(X, Y, order=2, X_test=X_test)
# ridge_poly_regression(X, Y, LAMBDA=1, order=2, form='auto', X_test=X_test)


'''
used for feature selection, to see which X features have high correlation with Y, to reduce dimension of X
    option 1: pick k features with highest pearson correlation values
    option 2: pick features with pearson correlation values above a threshold c
    k and c are magic numbers decided outside of this function
'''
pearson_correlation(X,Y)

# print("did you add offset for X if you are using linear regression? and DON'T use offset for polynomial Regression!" )
