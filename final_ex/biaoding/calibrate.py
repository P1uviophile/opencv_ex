import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
square_size_mm = 11  # 每个格子的宽度（单位：毫米）

# 计算物体点坐标
objp = np.zeros((24 * 13, 3), np.float32)
objp[:, :2] = np.mgrid[0:24, 0:13].T.reshape(-1, 2) * square_size_mm
# print(objp)

objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in data plane.

images = glob.glob('cali/*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (24, 13), None)
    # print(ret)
    # If found, add object points, data points (after refining them)
    if ret:
        objpoints.append(objp)
        cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (24, 13), corners, ret)
        # cv.imshow('img', img)
        # cv.waitKey(0) & 0xFF

cv.destroyAllWindows()

# 函数返回 整体RMS重投影误差, 相机矩阵, 畸变系数, 旋转和平移向量等
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error / len(objpoints)))

np.savez('B.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
