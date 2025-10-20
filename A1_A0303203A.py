import numpy as np


# Please replace "MatricNumber" with your actual matric number here and in the filename
def A1_A0303203A(X, y):
    """
    Input type
    :X type: numpy.ndarray
    :y type: numpy.ndarray

    Return typepy
    :InvXTX type: numpy.ndarray
    :w type: numpy.ndarray
   
    """

    # your code goes here
    XTX = np.dot(X.T, X)
    InvXTX = np.linalg.inv(XTX) 
    w = np.dot(InvXTX, np.dot(X.T, y))

    # return in this order
    return InvXTX, w
