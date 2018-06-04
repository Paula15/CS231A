import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.io as sio
from epipolar_utils import *

'''
LLS_EIGHT_POINT_ALG  computes the fundamental matrix from matching points using 
linear least squares eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the linear least squares eight
point algorithm works
'''
def lls_eight_point_alg(points1, points2):
    # TODO: Implement this method!
    N = len(points1)
    W = np.zeros([N, 9])
    b = np.zeros([N])
    for i in range(N):
        ui, vi = points1[i][:2]
        ui_, vi_ = points2[i][:2]
        W[i] = [ui*ui_, vi*ui_, ui_, ui*vi_, vi*vi_, vi_, ui, vi, 1.0]
    U, s, VT = np.linalg.svd(W)
    F = VT[-1].reshape([3, 3])
    U, s, VT = np.linalg.svd(F)
    s = np.diag([s[0], s[1], 0.0])
    return np.dot(np.dot(U, s), VT)

'''
NORMALIZED_EIGHT_POINT_ALG  computes the fundamental matrix from matching points
using the normalized eight point algorithm
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    F - the fundamental matrix such that (points2)^T * F * points1 = 0
Please see lecture notes and slides to see how the normalized eight
point algorithm works
'''
def normalized_eight_point_alg(points1, points2):
    # TODO: Implement this method!
    def get_mean_dist(points):
        N = len(points)
        centroid = np.mean(points, axis=0)
        mean_dist = 0.0
        for p in points: mean_dist += np.sum((p - centroid)**2)**0.5
        mean_dist /= N
        return centroid, mean_dist
    
    def get_T_matrix(points):
        centroid, mean_dist = get_mean_dist(points)
        T = np.identity(3)
        T[0, 2] = -centroid[0]
        T[1, 2] = -centroid[1]
        scale = [2.0/mean_dist, 2.0/mean_dist, 1.0]
        T = np.dot(np.diag(scale), T)
        return T

    def get_normalized_points(T, points):
        normalized_points = []
        for p in points:
            q = np.dot(T, p)
            normalized_points.append(q)
        return normalized_points

    T1, T2 = get_T_matrix(points1), get_T_matrix(points2)
    pts1, pts2 = get_normalized_points(T1, points1), \
                 get_normalized_points(T2, points2)
    Fq = lls_eight_point_alg(pts1, pts2)
    F = np.dot(np.dot(T2.T, Fq), T1)
    return F

'''
PLOT_EPIPOLAR_LINES_ON_IMAGES given a pair of images and corresponding points,
draws the epipolar lines on the images
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    im1 - a HxW(xC) matrix that contains pixel values from the first image 
    im2 - a HxW(xC) matrix that contains pixel values from the second image 
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    Nothing; instead, plots the two images with the matching points and
    their corresponding epipolar lines. See Figure 1 within the problem set
    handout for an example
'''
def plot_epipolar_lines_on_images(points1, points2, im1, im2, F):
    # TODO: Implement this method!
    def init():
        fig, axs = plt.subplots(1, 2)
        for ax, im in zip(axs, [im1, im2]):
            plt.axes(ax)
            ax.set_xlim([0, im.shape[1]])
            ax.set_ylim([im.shape[0], 0])
            plt.imshow(im, cmap='gray', interpolation='bicubic', aspect='auto')
        return fig, axs

    def plot(ax, p, line):
        """
        line: [a, b, c]
        -> ax + by + c = 0
        -> y = -a/b*x - c/b
        """
        plt.axes(ax)
        a, b, c = line
        X = np.array(ax.get_xlim())
        Y = -c/b - X*a/b
        plt.plot(X, Y, '--', linewidth=1)
        plt.plot(p[0], p[1], 'o')

    fig, axs = init()
    for p1, p2 in zip(points1, points2):
        l1 = np.dot(F.T, p2)
        l2 = np.dot(F, p1)
        plot(axs[0], p1, l1)
        plot(axs[1], p2, l2)

'''
COMPUTE_DISTANCE_TO_EPIPOLAR_LINES  computes the average distance of a set a 
points to their corresponding epipolar lines
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the fundamental matrix such that (points2)^T * F * points1 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    average_distance - the average distance of each point to the epipolar line
'''
def compute_distance_to_epipolar_lines(points1, points2, F):
    # TODO: Implement this method!
    def get_dist(p, line):
        a, b, c = line
        return abs(np.dot(p, line)) / (a*a + b*b)**0.5

    mean_dist = 0.0
    for p1, p2 in zip(points1, points2):
        l1 = np.dot(F.T, p2)
        l2 = np.dot(F, p1)
        dist1 = get_dist(p1, l1)
        dist2 = get_dist(p2, l2)
        mean_dist += dist1
        mean_dist += dist2
    mean_dist /= (len(points1) + len(points2))
    return mean_dist

if __name__ == '__main__':
    for im_set in ['data/set1', 'data/set2']:
        print '-'*80
        print "Set:", im_set
        print '-'*80

        # Read in the data
        im1 = imread(im_set+'/image1.jpg')
        im2 = imread(im_set+'/image2.jpg')
        points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
        points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
        assert (points1.shape == points2.shape)

        # Running the linear least squares eight point algorithm
        F_lls = lls_eight_point_alg(points1, points2)
        print "Fundamental Matrix from LLS  8-point algorithm:\n", F_lls
        print "Distance to lines in image 1 for LLS:", \
            compute_distance_to_epipolar_lines(points1, points2, F_lls)
        print "Distance to lines in image 2 for LLS:", \
            compute_distance_to_epipolar_lines(points2, points1, F_lls.T)

        # Running the normalized eight point algorithm
        F_normalized = normalized_eight_point_alg(points1, points2)

        pFp = [points2[i].dot(F_normalized.dot(points1[i])) 
            for i in xrange(points1.shape[0])]
        print "p'^T F p =", np.abs(pFp).max()
        print "Fundamental Matrix from normalized 8-point algorithm:\n", \
            F_normalized

        print "Distance to lines in image 1 for normalized:", \
            compute_distance_to_epipolar_lines(points1, points2, F_normalized)
        print "Distance to lines in image 2 for normalized:", \
            compute_distance_to_epipolar_lines(points2, points1, F_normalized.T)

        # Plotting the epipolar lines
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_lls)
        plot_epipolar_lines_on_images(points1, points2, im1, im2, F_normalized)

        plt.show()
