# CS231A Homework 0, Problem 4
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc


def main():
    # ===== Problem 4a =====
    # Read in image1 as a grayscale image. Take the singular value
    # decomposition of the image.

    # BEGIN YOUR CODE HERE
    X = misc.imread('./image1.jpg', mode='L')
    P, D, Q = np.linalg.svd(X, full_matrices=False)
    # END YOUR CODE HERE

    # ===== Problem 4b =====
    # Save and display the best rank 1 approximation 
    # of the (grayscale) image1.

    # BEGIN YOUR CODE HERE
    D_1 = np.diag(np.concatenate([D[:1], np.zeros(len(D) - 1)]))
    X_approx = np.dot(np.dot(P, D_1), Q)
    misc.imsave('runs/4b.png', X_approx)
    # END YOUR CODE HERE

    # ===== Problem 4c =====
    # Save and display the best rank 20 approximation
    # of the (grayscale) image1.

    # BEGIN YOUR CODE HERE
    D_20 = np.diag(np.concatenate([D[:20], np.zeros(len(D) - 20)]))
    X_approx = np.dot(np.dot(P, D_20), Q)
    misc.imsave('runs/4c.png', X_approx)
    # END YOUR CODE HERE


if __name__ == '__main__':
    main()