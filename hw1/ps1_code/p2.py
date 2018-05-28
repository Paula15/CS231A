# CS231A Homework 1, Problem 2
import numpy as np

'''
DATA FORMAT

In this problem, we provide and load the data for you. Recall that in the original
problem statement, there exists a grid of black squares on a white background. We
know how these black squares are setup, and thus can determine the locations of
specific points on the grid (namely the corners). We also have images taken of the
grid at a front image (where Z = 0) and a back image (where Z = 150). The data we
load for you consists of three parts: real_XY, front_image, and back_image. For a
corner (0,0), we may see it at the (137, 44) pixel in the front image and the
(148, 22) pixel in the back image. Thus, one row of real_XY will contain the numpy
array [0, 0], corresponding to the real XY location (0, 0). The matching row in
front_image will contain [137, 44] and the matching row in back_image will contain
[148, 22]
'''

'''
COMPUTE_CAMERA_MATRIX
Arguments:
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    camera_matrix - The calibrated camera matrix (3x4 matrix)
'''
def compute_camera_matrix(real_XY, front_image, back_image):
    # TODO: Fill in this code
    n = len(real_XY)
    A = np.zeros([2 * n, 4])
    bx = np.zeros([2 * n])
    by = np.zeros([2 * n])

    for i in range(n):
        X, Y = real_XY[i]
        A[i] = [X, Y, 0, 1]
        bx[i] = front_image[i][0]
        by[i] = front_image[i][1]
    for i in range(n):
        X, Y = real_XY[i]
        A[i + n] = [X, Y, 150, 1]
        bx[i + n] = back_image[i][0]
        by[i + n] = back_image[i][1]

    mx, res_x, rank_x, s_x = np.linalg.lstsq(A, bx)
    my, res_y, rank_y, s_y = np.linalg.lstsq(A, by)

    M = np.zeros([3, 4])
    M[0] = mx
    M[1] = my
    M[2] = [0, 0, 0, 1]
    return M

'''
RMS_ERROR
Arguments:
     camera_matrix - The camera matrix of the calibrated camera
     real_XY - Each row corresponds to an actual point on the 2D plane
     front_image - Each row is the pixel location in the front image where Z=0
     back_image - Each row is the pixel location in the back image where Z=150
Returns:
    rms_error - The root mean square error of reprojecting the points back
                into the images
'''
def rms_error(camera_matrix, real_XY, front_image, back_image):
    #TODO: Fill in this code
    n = len(real_XY)
    XYZ1_0 = np.r_[real_XY.T, np.zeros((1, n)), np.ones((1, n))]
    XYZ1_150 = np.r_[real_XY.T, np.ones((1, n)) * 150, np.ones((1, n))]
    xy1_0 = np.matmul(camera_matrix, XYZ1_0)[:2, :].T
    xy1_150 = np.matmul(camera_matrix, XYZ1_150)[:2, :].T
    err = 0.0
    err += np.sum((front_image - xy1_0) ** 2)
    err += np.sum((back_image - xy1_150) ** 2)
    err = np.sqrt(err / float(n))
    return err


if __name__ == '__main__':
    # Loading the example coordinates setup
    real_XY = np.load('npy/real_XY.npy')
    front_image = np.load('npy/front_image.npy')
    back_image = np.load('npy/back_image.npy')

    camera_matrix = compute_camera_matrix(real_XY, front_image, back_image)
    print("Camera Matrix:\n", camera_matrix)
    print()
    print("RMS Error: ", rms_error(camera_matrix, real_XY, front_image, back_image))
