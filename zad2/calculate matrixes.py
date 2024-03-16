import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import random
import time

# create matrixes with random numbers of size NxN

N = 512
M = 512
def create_matrix(N, M, binarize=False):
    matrix = np.random.rand(N, M)
    if binarize:
        matrix = (matrix > 0.5).astype(int)
    return matrix


# A - base matrix
A = create_matrix(N, M, True)
# add row with values 1
A = np.vstack([A, np.ones(A.shape[1]).reshape(1, M)])
A = A.T
print("A=",A)

B = create_matrix(N, M, True)
# add row with values 1
B = B.T
# print("X=",X)
print("B=",B)


# minimize X so that A*X = B
X = np.linalg.lstsq(A, B, rcond=None)[0]

print("X=",X)

# compute error
error = np.linalg.norm(A.dot(X) - B)
print("error=",error)
