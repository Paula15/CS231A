# CS231A Homework 1, Problem 3
import numpy as np
from utils import mat2euler
import math

'''
COMPUTE_VANISHING_POINTS
Arguments:
    points - a list of all the points where each row is (x, y). Generally,
            it will contain four points: two for each parallel line.
            You can use any convention you'd like, but our solution uses the
            first two rows as points on the same line and the last
            two rows as points on the same line.
Returns:
    vanishing_point - the pixel location of the vanishing point
'''
def compute_vanishing_point(points):
    #TODO: Fill in this code
    points = np.c_[points, np.ones((4, 1))] # 转化为齐次坐标
    P0, P1, P2, P3 = points
    n1 = np.cross(P0, P1)   # 第一条直线的法向量
    n2 = np.cross(P2, P3)   # 第二条直线的法向量
    V = np.cross(n1, n2)    # 两条直线的交点
    V /= V[2]               # 将齐次坐标的最后一维归一
    return V

'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_points - a list of vanishing points

Returns:
    K - the intrinsic camera matrix (3x3 matrix)
'''
def compute_K_from_vanishing_points(vanishing_points):
    #TODO: Fill in this code
    """
    算法：
    设V1, V2, V3是3对相互垂直的平行线（方向d1, d2, d3）的3个消失点，则有：
    1) Vi = Kdi，K为待求的相机矩阵
    2) Vi.T W Vj = 0, i != j。其中W = (KK.T)^-1（因为di dj = 0）
    3) 假设方像素、无扭曲，则相机矩阵K有3个自由度，满足形式
        K = | α  0  cx |
            | 0  α  cy |
            | 0  0  1  |
       从而W = (KK.T)^-1满足形式
        W = | w1 0  w4 |
            | 0  w1 w5 |
            | w4 w5 w6 |
       其中w1, w4, w6待求，根据2)的3个约束条件确定
    4) 推导vi.T W Vj = 0的方程形式：
        Vi = [a, b, 1]
        Vj = [x, y, 1]
       有Vi.T W Vj = [xa + yb, x + a, y + b, 1] | w1 |
                                                | w4 |
                                                | w5 |
                                                | w6 |
       将i, j从0~2循环，可得3个方程，最终得到方程组
       A(3, 4) w(4, 1) = 0(3, 1)
    5) 此时方程个数多于未知数个数。
       解1：
           实际上，容易发现必有w6 = 1，因为本来K就只有3个自由度，
           从而可以直接给A加一个条件，令w6 = 1；
       解2：
           设A(3, 4)的奇异值分解为
            A = [u1, u2, u3] | a        0 | | v1.T |
                             |    b     0 | | v2.T |
                             |       c  0 | | v3.T |
                                            | v4.T |
           由于(v1, v2, v3, v4)正交，故
           有Av4  = [u1, u2, u3] | a        0 | | 0 | = 0
                                 |    b     0 | | 0 |
                                 |       c  0 | | 0 |
                                                | 1 |
           所以可取v4为一个解。
           （查了github上2份答案都这么做的，但并不明白原理……）
    6) 根据W，使用Cholesky分解求得K = cholesky(W^-1)即可
    """
    A = np.zeros([4, 4])
    for row, (i, j) in enumerate([(0, 1), (0, 2), (1, 2)]):
        Pi = vanishing_points[i] / vanishing_points[i][2]
        Pj = vanishing_points[j] / vanishing_points[i][2]
        a, b, _ = Pi
        x, y, _ = Pj
        A[row] = [x*a + y*b, x + a, y + b, 1.]
    # 限制w6 = 1
    A[3] = [0., 0., 0., 1.]     
    w1, w4, w5, w6 = np.linalg.solve(A, [0., 0., 0., 1.])
    W = np.array([[w1, 0., w4],
                  [0., w1, w5],
                  [w4, w5, w6]])
    # 注意：必须先Cholesky分解再求逆，否则结果相差巨大
    K = np.linalg.cholesky(W)   
    K = np.linalg.inv(K).T  
    return K

'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_pair1 - a list of a pair of vanishing points computed from lines within the same plane
    vanishing_pair2 - a list of another pair of vanishing points from a different plane than vanishing_pair1
    K - the camera matrix used to take both images

Returns:
    angle - the angle in degrees between the planes which the vanishing point pair comes from2
'''
def compute_angle_between_planes(vanishing_pair1, vanishing_pair2, K):
    #TODO: Fill in this code
    P0, P1 = vanishing_pair1
    P2, P3 = vanishing_pair2
    K_inv = np.linalg.inv(K)
    d0, d1 = np.dot(K_inv, P0), np.dot(K_inv, P1)
    d2, d3 = np.dot(K_inv, P2), np.dot(K_inv, P3)
    n1 = np.cross(d0, d1)
    n2 = np.cross(d2, d3)
    cos = np.dot(n1, n2) / (np.sum(n1 ** 2) * np.sum(n2 ** 2)) ** 0.5
    angle = math.acos(cos) * 180 / math.pi
    return angle

'''
COMPUTE_K_FROM_VANISHING_POINTS
Arguments:
    vanishing_points1 - a list of vanishing points in image 1
    vanishing_points2 - a list of vanishing points in image 2
    K - the camera matrix used to take both images

Returns:
    R - the rotation matrix between camera 1 and camera 2
'''
def compute_rotation_matrix_between_cameras(vanishing_points1, vanishing_points2, K):
    #TODO: Fill in this code
    """
    算法：
    1) 设d1, d2分别是相机1, 2拍摄的同一平行线的3D方向向量，则有d2 = Rd1
       因为有p = K[R T]P，因此可知旋转矩阵的定义是P' = RP（P为3D），即在
       旋转后的相机看来，世界中的每个点都做了变换R
    2) R [d1(0), d1(1), ..., d1(n-1)] = [d2(0), d2(1), ..., d2(n-1)]
       | d2(0).T   |     | d1(0).T   |
       | d2(1).T   | =   | d1(1).T   | R.T
       | ...       |     | ...       |
       | d2(n-1).T |     | d1(n-1).T |
    """
    K_inv = np.linalg.inv(K)
    A = np.dot(K_inv, vanishing_points1.T)
    B = np.dot(K_inv, vanishing_points2.T)
    R = np.linalg.lstsq(A.T, B.T)[0].T
    return R

if __name__ == '__main__':
    # Part A: Compute vanishing points
    v1 = compute_vanishing_point(np.array([[674,1826],[2456,1060],[1094,1340],[1774,1086]]))
    v2 = compute_vanishing_point(np.array([[674,1826],[126,1056],[2456,1060],[1940,866]]))
    v3 = compute_vanishing_point(np.array([[1094,1340],[1080,598],[1774,1086],[1840,478]]))

    v1b = compute_vanishing_point(np.array([[314,1912],[2060,1040],[750,1378],[1438,1094]]))
    v2b = compute_vanishing_point(np.array([[314,1912],[36,1578],[2060,1040],[1598,882]]))
    v3b = compute_vanishing_point(np.array([[750,1378],[714,614],[1438,1094],[1474,494]]))

    # Part B: Compute the camera matrix
    vanishing_points = [v1, v2, v3]
    print("Intrinsic Matrix:\n",compute_K_from_vanishing_points(vanishing_points))

    K_actual = np.array([[2448.0, 0, 1253.0],[0, 2438.0, 986.0],[0,0,1.0]])
    print()
    print("Actual Matrix:\n", K_actual)

    # Part D: Estimate the angle between the box and floor
    floor_vanishing1 = v1
    floor_vanishing2 = v2
    box_vanishing1 = v3
    box_vanishing2 = compute_vanishing_point(np.array([[1094,1340],[1774,1086],[1080,598],[1840,478]]))
    angle = compute_angle_between_planes([floor_vanishing1, floor_vanishing2], [box_vanishing1, box_vanishing2], K_actual)
    print()
    print("Angle between floor and box:", angle)

    # Part E: Compute the rotation matrix between the two cameras
    rotation_matrix = compute_rotation_matrix_between_cameras(np.array([v1, v2, v3]), np.array([v1b, v2b, v3b]), K_actual)
    print()
    print("Rotation between two cameras:\n", rotation_matrix)
    z,y,x = mat2euler(rotation_matrix)
    print
    print("Angle around z-axis (pointing out of camera): %f degrees" % (z * 180 / math.pi))
    print("Angle around y-axis (pointing vertically): %f degrees" % (y * 180 / math.pi))
    print("Angle around x-axis (pointing horizontally): %f degrees" % (x * 180 / math.pi))
