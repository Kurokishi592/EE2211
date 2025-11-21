from LinearRegression import linear_regression
from PolynomialRegression import polynomial_regression
from RidgePolynomialRegression import ridge_poly_regression
from RidgeRegression import ridge_regression
from OneHotLinearClassification import onehot_linearclassification
from pearson_correlation import pearson_correlation
from GradientDescent import GradientDescent
from regression_tree_house import regression_tree_house
from TreeRegressor import TreeRegressor
from TreeClassifier import TreeClassifier
from ManualTreeRegressor import manual_tree_regressor
from AutoTreeRegressor import auto_tree_regressor
from ManualTreeClassifier import manual_tree_classifier
from AutoTreeClassifier import auto_tree_classifier
from k_means_cluster import custom_kmeans
from k_means_cluster_lib import kmeans_sklearn
import numpy as np
from sklearn.metrics import mean_squared_error
from EnterMetrics import EnterMetrics

''' 
no need to add column of 1s to X for regression. X_fitted does it already. First row of w is w0.
for correlation, row is sample, column is feature. Comparing each feature column to one target Y
'''
X=np.array(
    [[3.3459, 2.7435, -1.7253],
     [1.0893,2.9113,-0.7804],
     [3.2103,1.4706,-0.9944],
     [1.744,1.2895,0.5307],
     [1.6762,2.1366,-1.0502]
   ]
);

''' 
for multi-class classification (poly, ridge, ridgepoly), manually key in the onehot encoding: class 0 = [1, 0, 0], class 1 = [0, 1, 0], class 2 = [0, 0, 1] ...
    only the function "onehot_linear" has auto one hot encoding, simply type the class number 0, 1, 2 ...etc 
    this code's argmax starts from class 0 not 1. 
for binary, all functions are the same, key in -1 or 1 will do.
for correlation, row is samples. should be comparing to one target only so only 1 column
'''
Y=np.array(
    [[2.9972], 
     [1.1399], 
     [2.228], 
     [0.3387], 
     [2.5042]
    ]
);

''' same dont add one column of 1s to X_test for regression'''
X_test=np.array(
    [[6]
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
# polynomial_regression(X, Y, order=4, X_test=X_test)
# ridge_poly_regression(X, Y, LAMBDA=1, order=2, form='auto', X_test=X_test)

'''
used for feature selection, to see which X features have high correlation with Y, to reduce dimension of X
    option 1: pick k features with highest pearson correlation values
    option 2: pick features with pearson correlation values above a threshold c
    k and c are magic numbers decided outside of this function
'''
# pearson_correlation(X,Y)

''' ----------------------------------------------------------------------------------------------------------------------------------'''

''' 
classification task and regression task (1D to 1D regression only) using decision trees 
regression_tree_house is just an example code, with custom tree regressor
    X_train and y_train to train sklearn decision tree 
    X_test and y_test to evaluate the trained decision tree
    criterion for regressor: 'squared_error' (default)
    criterion for classifier by impurity: 'gini' or 'entropy' or 'misclassification'
    max_depth to prune tree (reduce complexity and overfitting)
    Nested per-node thresholds: [root, [depth1_left, depth1_right], [depth2_n0, depth2_n1, depth2_n2, depth2_n3], ...]
    Provide enough thresholds to drive splits; missing entries stop growth for that node.
both function returns nothing, just print out the training and test accuracies/MSEs
'''
X_train = np.array([0.2, 0.7, 1.8, 2.2, 3.7, 4.1, 4.5, 5.1, 6.3, 7.4])
y_train = np.array([2.1, 1.5, 5.8, 6.1, 9.1, 9.5, 9.8, 12.7, 13.8, 15.9])
X_test = np.array([1.0, 3.0, 6.0]) # leave untouched if not used
y_test = np.array([0, 1, 2]) # leave untouched if not used
max_depth = 20
impurity = 'gini' 
decision_threshold = [3.0, [1.0, 4.8], [0.5, 1.5, 4.0, 5.4]]

# manual_tree_regressor(X_train, y_train, max_depth=max_depth, decision_threshold=decision_threshold, X_test=X_test)
# auto_tree_regressor(X_train, y_train, max_depth=max_depth, X_test=X_test)
# manual_tree_classifier(X_train, y_train, max_depth=max_depth, decision_threshold=decision_threshold, X_test=X_test, criterion=impurity)
# auto_tree_classifier(X_train, y_train, max_depth=max_depth, X_test=X_test, criterion=impurity)

# TreeClassifier(X_train, X_test, y_train, y_test, criterion='gini', max_depth=3)
# TreeRegressor(X_train, X_test, y_train, y_test, criterion='squared_error', max_depth=3)
# regression_tree_house()

'''
perform gradient descent (for multivariable functions)
GradientDescent(f, f_prime, initial, learning_rate, num_iters)
    use lambda functions for f and f_prime, parameters: function (xyz: [0] refers to x, [1] refers to y, [2] refers to z ...)
    initial: initial values of all variables as a tuple
    [0]: steps at each iteration
    [1]: function values at each iteration
    [2]: gradient vectors at each iteration
if multiple variables, put PARTIAL derivative of each variable in f_prime return tuple
(x,y,z) => (df/dx, df/dy, df/dz)
'''
learning_rate = 0.2
num_iters = 3

print("Values of parameters at each step (first row is initial values): \n")
print(GradientDescent(lambda xy:xy[0]**2 + xy[0]*xy[1]**2, lambda xy:(2*xy[0] + xy[1]**2, 2*xy[0]*xy[1]), (3,2) , learning_rate, num_iters)[0], "\n")
print("Function values at each step: \n")
print(GradientDescent(lambda xy:xy[0]**2 + xy[0]*xy[1]**2, lambda xy:(2*xy[0] + xy[1]**2, 2*xy[0]*xy[1]), (3,2), learning_rate, num_iters)[1], "\n")
print("Gradient vectors (partial derivatives) at each step: \n")
print(GradientDescent(lambda xy:xy[0]**2 + xy[0]*xy[1]**2, lambda xy:(2*xy[0] + xy[1]**2, 2*xy[0]*xy[1]), (3,2), learning_rate, num_iters)[2], "\n")

# print("Values of parameters at each step (first row is initial values): \n")
# print(GradientDescent(lambda b:np.sin(b)**2, lambda b:2*np.sin(b)*np.cos(b), 3, learning_rate, num_iters)[0], "\n")
# print("Function values at each step: \n")
# print(GradientDescent(lambda b:np.sin(b)**2, lambda b:2*np.sin(b)*np.cos(b), 3, learning_rate, num_iters)[1], "\n")
# print("Gradient vectors (partial derivatives) at each step: \n")
# print(GradientDescent(lambda b:np.sin(b)**2, lambda b:2*np.sin(b)*np.cos(b), 3, learning_rate, num_iters)[2], "\n")


'''
perform kmeans clustering
returns converged centers and cluster labels. Auto stop iterations upon convergence.
for fuzzy means, run fuzzy_cmeans.py directly instead
'''
x1 = np.array([50])
x2 = np.array([60])
x3 = np.array([66])
x4 = np.array([68])
x5 = np.array([71])
x6 = np.array([72])
x7 = np.array([75])
x8 = np.array([82])
x9 = np.array([90])
x10 = np.array([99])
data_points = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
c1_init = x3.copy()
c2_init = x7.copy()
# c3_init = x7.copy()
centers_init = np.array([c1_init, c2_init])

# custom_kmeans(data_points, centers_init, n_clusters=2, max_iterations=100)