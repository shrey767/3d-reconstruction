import numpy as np
import helper as hlp
import skimage.io as io
import submission as sub
import matplotlib.pyplot as plt
import cv2 as cv
from submission import *
from helper import *
# 1. Load the two temple images and the points from data/some_corresp.npz

im1 = cv.imread("data/im1.png")
im2 = cv.imread("data/im2.png")
corresp = np.load("data/some_corresp.npz")
pts1 = corresp['pts1']
pts2 = corresp['pts2']
MAXH, MAXW, _ = im1.shape
# print(MAXH,MAXW)
# print(len(pts1))

# 2. Run eight_point to compute F

F  = eight_point(pts1,pts2,max(MAXH,MAXW))

# 3. Load points in image 1 from data/temple_coords.npz

im1_coords = np.load("data/temple_coords.npz")
pts1 = im1_coords['pts1']
# 4. Run epipolar_correspondences to get points in image 2

pts2 = epipolar_correspondences(im1=im1,im2=im2,F=F,pts1=pts1)

# 5. Compute the camera projection matrix P1

intrinsics = np.load("data/intrinsics.npz")
K1 = intrinsics['K1']
K2 = intrinsics['K2']
E = essential_matrix(F=F,K1=K1,K2=K2)

# 6. Use camera2 to get 4 camera projection matrices P2

P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))
M2s = camera2(E)

# 7. Run triangulate using the projection matrices

# 8. Figure out the correct P2

best_count = 0
best_P2 = None
best_points = None

for i in range(4):
    M2 = M2s[:, :, i]
    P2 = K2 @ M2
    points3D = triangulate(P1, pts1, P2, pts2)

    R = M2[:, :3]
    t = M2[:, 3]

    count = 0

    for X in points3D:

        # in front of camera 1
        cond1 = X[2] > 0

        # transform into camera 2 coordinates
        X2 = R @ X + t

        # in front of camera 2
        cond2 = X2[2] > 0

        if cond1 and cond2:
            count += 1
    # print(count)
    if count > best_count:
        best_count = count
        best_P2 = P2
        best_points = points3D

# 9. Scatter plot the correct 3D points

fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    best_points[:,0],   # X coordinates
    best_points[:,1],   # Y coordinates
    best_points[:,2]    # Z coordinates
)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()

# 10. Save the computed extrinsic parameters (R1,R2,t1,t2) to data/extrinsics.npz

M2 = np.linalg.inv(K2) @ best_P2
R2, t2 = M2[:, :3], M2[:, 3]
R1 = np.eye(3)
t1 = np.zeros(3)
np.savez("data/extrinsics.npz", R1=R1, t1=t1, R2=R2, t2=t2)