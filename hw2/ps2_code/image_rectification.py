#encoding=utf-8
import numpy as np
import matplotlib.pyplot as plt
from fundamental_matrix_estimation import *

'''
COMPUTE_EPIPOLE computes the epipole in homogenous coordinates
given matching points in two images and the fundamental matrix
Arguments:
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
    F - the Fundamental matrix such that (points1)^T * F * points2 = 0

    Both points1 and points2 are from the get_data_from_txt_file() method
Returns:
    epipole - the homogenous coordinates [x y 1] of the epipole in the image
'''
def compute_epipole(points1, points2, F):
    # TODO: Implement this method!
    N = len(points1)
    L = np.zeros([N, 3])
    for i, (p1, p2) in enumerate(zip(points1, points2)):
        l = np.dot(F.T, p2)
        L[i] = l
    U, s, VT = np.linalg.svd(L)
    e = VT[-1]
    e /= e[2]
    return e
    
'''
COMPUTE_MATCHING_HOMOGRAPHIES determines homographies H1 and H2 such that they
rectify a pair of images
Arguments:
    e2 - the second epipole
    F - the Fundamental matrix
    im2 - the second image
    points1 - N points in the first image that match with points2
    points2 - N points in the second image that match with points1
Returns:
    H1 - the homography associated with the first image
    H2 - the homography associated with the second image
'''
def compute_matching_homographies(e2, F, im2, points1, points2):
    # TODO: Implement this method!
    """
    两个大坑：
    1. 任何地方涉及齐次坐标，一定要将最后一维归一化！
    2. 计算H2时用到的e2都是前面已经算出来的中间结果，而不是e2的初始值。`
    
    算法：
    1. 计算H2
    H2 = T-1GRT
    T：将e2平移到中心位置
    R：将e2旋转到水平位置[f, 0, 1]
    G：将e2 = [f, 0, 1]映射到无穷远
    
    2. 计算H1
    H1 = HAH2M
    (2.1) M
        M = e_x F + e*[1, 1, 1]
        e_x = |0 -z y|
              |z 0 -x|
              |-y x 0|
    (2.2) HA
        HA = |a1 a2 a3|
             |0  0  0 |
             |0  0  0 |
        Wa = b
        p1_hat = H2Mp1
        p2_hat = H2p2
        W = | x1_hat[1] y1_hat[1] 1 |  b = | x2_hat[1] |
            | ...                   |      | ...       |
            | x1_hat[n] y1_hat[n] 1 |      | x2_hat[n] |
    """
    # 1. H2
    # (1.1) T
    H, W = im2.shape
    T = np.array([[1, 0, -W/2.], [0, 1, -H/2.], [0, 0, 1]])
    # (1.2) R
    e2_T = T.dot(e2)
    e2_T /= e2_T[2]
    alpha = 1 if e2_T[0] >= 0 else -1
    denom = np.sqrt(e2_T[0]**2 + e2_T[1]**2)
    R1 = alpha * e2_T[0] / denom
    R2 = alpha * e2_T[1] / denom
    R = np.array([[R1, R2, 0], [-R2, R1, 0], [0, 0, 1]])
    # (1.3) G
    e2_RT = R.dot(e2_T)
    e2_RT /= e2_RT[2]
    f = e2_RT[0]
    G = np.array([[1, 0, 0], [0, 1, 0], [-1/f, 0, 1]])
    # (1.4) H2
    H2 = np.linalg.inv(T).dot(G).dot(R).dot(T)
    """
    """
    # (2.1) M
    e_x = np.array([[0, -e2[2], e2[1]], [e2[2], 0, -e2[0]], [-e2[1], e2[0], 0]])
    M = np.dot(e_x, F) + np.outer(e2, [1, 1, 1])
    # (2.2) HA
    N = len(points1)
    W = np.zeros([N, 3])
    p1_hat = H2.dot(M).dot(points1.T)
    p2_hat = H2.dot(points2.T)
    p1_hat /= p1_hat[2]
    p2_hat /= p2_hat[2]
    p1_hat = p1_hat.T
    p2_hat = p2_hat.T
    W = np.c_[p1_hat[:, :2], np.ones([N, 1])]
    b = p2_hat[:, 0]
    a = np.linalg.lstsq(W, b)[0]
    HA = np.array([[a[0], a[1], a[2]], [0, 1, 0], [0, 0, 1]])
    H1 = HA.dot(H2).dot(M)
    return H1, H2

if __name__ == '__main__':
    # Read in the data
    im_set = 'data/set1'
    im1 = imread(im_set+'/image1.jpg')
    im2 = imread(im_set+'/image2.jpg')
    points1 = get_data_from_txt_file(im_set+'/pt_2D_1.txt')
    points2 = get_data_from_txt_file(im_set+'/pt_2D_2.txt')
    assert (points1.shape == points2.shape)

    F = normalized_eight_point_alg(points1, points2)
    e1 = compute_epipole(points1, points2, F)
    e2 = compute_epipole(points2, points1, F.transpose())
    print "e1", e1
    print "e2", e2

    # Find the homographies needed to rectify the pair of images
    H1, H2 = compute_matching_homographies(e2, F, im2, points1, points2)
    print "H1:\n", H1
    print
    print "H2:\n", H2

    # Transforming the images by the homographies
    new_points1 = H1.dot(points1.T)
    new_points2 = H2.dot(points2.T)
    new_points1 /= new_points1[2,:]
    new_points2 /= new_points2[2,:]
    new_points1 = new_points1.T
    new_points2 = new_points2.T
    rectified_im1, offset1 = compute_rectified_image(im1, H1)
    rectified_im2, offset2 = compute_rectified_image(im2, H2)
    new_points1 -= offset1 + (0,)
    new_points2 -= offset2 + (0,)

    # Plotting the image
    F_new = normalized_eight_point_alg(new_points1, new_points2)
    plot_epipolar_lines_on_images(new_points1, new_points2, rectified_im1, rectified_im2, F_new)
    plt.show()
