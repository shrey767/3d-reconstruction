"""
Homework 5
Submission Functions
"""

import numpy as np


"""
Q3.1.1 Eight Point Algorithm
       [I] pts1, points in image 1 (Nx2 matrix)
           pts2, points in image 2 (Nx2 matrix)
           M, scalar value computed as max(H1,W1)
       [O] F, the fundamental matrix (3x3 matrix)
"""
def eight_point(pts1, pts2, M):
    N = pts1.shape[0]

    # --- Proper Hartley normalization ---
    # Center each point cloud then scale so mean distance from origin is sqrt(2)
    mean1 = pts1.mean(axis=0)          # (cx, cy) for image 1
    mean2 = pts2.mean(axis=0)          # (cx, cy) for image 2
    scale1 = np.sqrt(2) / np.std(pts1 - mean1)
    scale2 = np.sqrt(2) / np.std(pts2 - mean2)

    # Build 3x3 normalization matrices
    T1 = np.array([
        [scale1, 0,      -scale1 * mean1[0]],
        [0,      scale1, -scale1 * mean1[1]],
        [0,      0,       1               ]
    ])
    T2 = np.array([
        [scale2, 0,      -scale2 * mean2[0]],
        [0,      scale2, -scale2 * mean2[1]],
        [0,      0,       1               ]
    ])

    # Apply normalization
    pts1_h = np.hstack([pts1, np.ones((N, 1))])   # (N,3)
    pts2_h = np.hstack([pts2, np.ones((N, 1))])
    pts1_n = (T1 @ pts1_h.T).T                    # (N,3) normalized
    pts2_n = (T2 @ pts2_h.T).T

    # Build A matrix from normalized points
    A = np.zeros((N, 9))
    for i in range(N):
        x,  y  = pts1_n[i, :2]
        xp, yp = pts2_n[i, :2]
        A[i] = [xp*x, xp*y, xp,
                yp*x, yp*y, yp,
                x,    y,    1 ]

    # Solve Af = 0 via SVD
    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt

    # Denormalize: F_original = T2^T @ F_normalized @ T1
    F = T2.T @ F @ T1

    return F


"""
Q3.1.2 Epipolar Correspondences
       [I] im1, image 1 (H1xW1 matrix)
           im2, image 2 (H2xW2 matrix)
           F, fundamental matrix from image 1 to image 2 (3x3 matrix)
           pts1, points in image 1 (Nx2 matrix)
       [O] pts2, points in image 2 (Nx2 matrix)
"""
def epipolar_correspondences(im1, im2, F, pts1):
    window = 11        # larger window → more reliable patch matching
    half   = window // 2

    pts2 = []
    h, w = im2.shape[:2]

    for p in pts1:
        x1, y1 = int(p[0]), int(p[1])

        # Epipolar line in image 2:  a*x + b*y + c = 0
        l = F @ np.array([x1, y1, 1.0])
        a, b, c = l

        # Template patch from image 1
        patch1 = im1[y1-half : y1+half+1,
                     x1-half : x1+half+1].astype(float)

        best_error = np.inf
        best_point = None

        # Choose iteration axis to avoid dividing by a near-zero coefficient.
        # Temple cameras are side-by-side → lines are roughly horizontal → |b| > |a|
        # → iterate over x, solve y = -(a*x + c) / b
        if abs(b) >= abs(a):
            for x2 in range(half, w - half):
                y2 = int(round(-(a * x2 + c) / b))
                if y2 < half or y2 >= h - half:
                    continue
                patch2 = im2[y2-half : y2+half+1,
                             x2-half : x2+half+1].astype(float)
                if patch2.shape != patch1.shape:
                    continue
                error = np.sum((patch1 - patch2) ** 2)
                if error < best_error:
                    best_error = error
                    best_point = [x2, y2]
        else:
            # Nearly vertical line → iterate over y, solve x = -(b*y + c) / a
            for y2 in range(half, h - half):
                x2 = int(round(-(b * y2 + c) / a))
                if x2 < half or x2 >= w - half:
                    continue
                patch2 = im2[y2-half : y2+half+1,
                             x2-half : x2+half+1].astype(float)
                if patch2.shape != patch1.shape:
                    continue
                error = np.sum((patch1 - patch2) ** 2)
                if error < best_error:
                    best_error = error
                    best_point = [x2, y2]

        pts2.append(best_point)

    return np.array(pts2)


"""
Q3.1.3 Essential Matrix
       [I] F, the fundamental matrix (3x3 matrix)
           K1, camera matrix 1 (3x3 matrix)
           K2, camera matrix 2 (3x3 matrix)
       [O] E, the essential matrix (3x3 matrix)
"""
def essential_matrix(F, K1, K2):
    return K2.T @ F @ K1


"""
Q3.1.4 Triangulation
       [I] P1, camera projection matrix 1 (3x4 matrix)
           pts1, points in image 1 (Nx2 matrix)
           P2, camera projection matrix 2 (3x4 matrix)
           pts2, points in image 2 (Nx2 matrix)
       [O] pts3d, 3D points in space (Nx3 matrix)
"""
def triangulate(P1, pts1, P2, pts2):
    N = pts1.shape[0]
    points3D = np.zeros((N, 3))

    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        A = np.array([
            x1 * P1[2] - P1[0],
            y1 * P1[2] - P1[1],
            x2 * P2[2] - P2[0],
            y2 * P2[2] - P2[1]
        ])

        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]
        points3D[i] = X[:3]

    return points3D


def rectify_pair(K1, K2, R1, R2, t1, t2):
    c1 = -np.linalg.inv(R1) @ t1
    c2 = -np.linalg.inv(R2) @ t2
    r1 = (c1-c2)/np.linalg.norm(c1-c2)
    r2 = -np.cross(R1[2,:].T,r1)
    r3 = -np.cross(r1,r2)
    R_ = np.array([r1,r2,r3]).T
    R1p=R_
    R2p=R_
    K1p = K2
    K2p = K2
    t1p = -(R1p @ c1)
    t2p = -(R2p @ c2)
    M1 = (K1p @ R1p) @ np.linalg.inv(K1 @ R1)
    M2 = (K2p @ R2p) @ np.linalg.inv(K2 @ R2)
    return M1,M2,K1p,K2p,R1p,R2p,t1p,t2p

import numpy as np
from scipy.signal import convolve2d

def get_disparity(im1, im2, max_disp, win_size):

    h, w = im1.shape

    dispM = np.zeros((h, w))

    half = win_size // 2

    # window summation filter
    kernel = np.ones((win_size, win_size))

    best_cost = np.full((h, w), np.inf)

    for d in range(max_disp + 1):

        # shift image2 to the right
        shifted = np.roll(im2, d, axis=1)

        # invalid region after shifting
        shifted[:, :d] = 0

        # squared difference
        diff = (im1 - shifted) ** 2

        # sum SSD over window
        cost = convolve2d(diff, kernel, mode='same')

        # update best disparities
        mask = cost < best_cost

        best_cost[mask] = cost[mask]

        dispM[mask] = d

    return dispM


def get_depth(dispM, K1, K2, R1, R2, t1, t2):

    # camera centers
    c1 = -R1.T @ t1
    c2 = -R2.T @ t2

    # baseline
    b = np.linalg.norm(c1 - c2)

    # focal length
    f = K1[0,0]

    depthM = np.zeros(dispM.shape)

    mask = dispM > 0

    depthM[mask] = (b * f) / dispM[mask]

    return depthM


def estimate_pose(x, X):
    pass

def estimate_params(P):
    pass
