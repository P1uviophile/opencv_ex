import glob
import cv2
import numpy as np
import cv2 as cv


def usePnP(scenePoints):
    # 参考物体在实际空间中的位置
    objectPoints = np.array([[0, 0, 0],
                             [240, 0, 0],
                             [240, 135, 0],
                             [0, 135, 0]], dtype=np.float32)
    # 计算rvec、tvec
    ret, rvec, tvec = cv2.solvePnP(objectPoints, np.array(scenePoints), mtx, dist,
                                   flags=cv2.SOLVEPNP_EPNP)
    if ret:
        print("solvePnP successfully!")
    else:
        print("solvePnP unsuccessfully…check your data again")
        rvec = np.array([0, 0, 0], dtype=np.float32)
        tvec = np.array([0, 0, 0], dtype=np.float32)

    return rvec, tvec


def main(fname, index=1):
    imgScene = cv2.imread(fname)
    # imgScene = cv2.resize(imgScene, dsize=None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)  # 图像过大就缩小了再处理

    if imgObject is None or imgScene is None:
        print("Error reading images")
        return

    # 创建SIFT对象
    sift = cv2.SIFT_create()

    # 检测特征点并计算特征描述子
    keypointsObject, descriptorsObject = sift.detectAndCompute(imgObject, None)
    keypointsScene, descriptorsScene = sift.detectAndCompute(imgScene, None)

    # 使用FLANN法进行匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptorsObject, descriptorsScene, k=2)

    goodMatches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            goodMatches.append(m)

    # 显示匹配结果
    imgMatches = cv2.drawMatches(imgObject, keypointsObject, imgScene, keypointsScene, goodMatches, None)
    # cv2.imshow("Matches", imgMatches)

    # 使用findHomography找出相应的透视变换
    objPoints = np.float32([keypointsObject[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    scenePoints = np.float32([keypointsScene[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(objPoints, scenePoints, cv2.RANSAC, 5.0)

    # 映射点群，在场景中获取目标位置
    objCorners = np.float32(
        [[0, 0], [imgObject.shape[1], 0], [imgObject.shape[1], imgObject.shape[0]], [0, imgObject.shape[0]]]).reshape(
        -1, 1, 2)
    sceneCorners = cv2.perspectiveTransform(objCorners, H)

    # img = imgScene
    # 显示检测结果
    # for i in range(4):
    # cv2.line(img, tuple(sceneCorners[i][0]), tuple(sceneCorners[(i + 1) % 4][0]), (0, 255, 0), 3)
    # img = cv2.resize(img, dsize=None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)  # 图像过大就缩小了再处理
    # print(sceneCorners)
    # cv2.imshow("Detection Result", img)

    # 计算位置姿态
    rvec, tvec = usePnP(sceneCorners.reshape(-1, 2))
    print("rvec:", rvec)
    print("tvec:", tvec)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    axis = np.float32(vertices)

    # project 3D points to image plane
    imgpts, jac = cv.projectPoints(axis, rvec, tvec, mtx, dist)
    img = draw(imgScene, imgpts)
    imgpts, jac = cv.projectPoints(axis_qu, rvec, tvec, mtx, dist)
    img = draw_qu(img, imgpts)
    # img = cv2.resize(img, dsize=None, fx=0.2, fy=0.2, interpolation=cv2.INTER_LINEAR)  # 图像过大就缩小了再处理
    # cv.imshow('img', img)
    file = r"camera_parameters\\out\\" + str(index) + ".jpg"
    cv.imwrite(file, img)
    return rvec, tvec


def draw_qu(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), 255, 3)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img


def draw(img, imgpts):
    # 创建一个与原图大小相同的掩膜，数据类型为无符号8位整数
    zeros = np.zeros(img.shape, dtype=np.uint8)
    mask_img = img

    color_idx = 0  # 颜色索引
    # 遍历字典中的每个形状
    for face in faces:
        # 从 face 中获取三个顶点的索引
        pt1_idx, pt2_idx, pt3_idx = face
        # 获取三个顶点的坐标
        pts = np.array([imgpts[pt1_idx], imgpts[pt2_idx], imgpts[pt3_idx]], np.int32)
        # 使用fillPoly填充多边形
        mask = cv2.fillPoly(zeros, [pts], color=colors[color_idx])
        color_idx = (color_idx + 1) % len(colors)
        # 将掩膜与原始图像相乘，创建带有标记的图像
        mask_img = 0.1 * mask + mask_img

    imgpts = np.int32(imgpts).reshape(-1, 2)
    for face in faces:
        # 从 face 中获取三个顶点的索引
        pt1_idx, pt2_idx, pt3_idx = face
        # 获取三个顶点的坐标
        pts = np.array([[imgpts[pt1_idx]], [imgpts[pt2_idx]], [imgpts[pt3_idx]]], np.int32)
        # 绘制三角形的边框
        mask_img = cv.polylines(mask_img, [pts], True, (0, 0, 0), 3)  # 使用 polylines 同时绘制边框和填充
        # 更新颜色索引，确保不会超出颜色列表的范围
        # color_idx = (color_idx + 1) % len(colors)
    return mask_img


def get_axis():
    m = np.sqrt(50 - 10 * np.sqrt(5)) / 10 * 30
    n = np.sqrt(50 + 10 * np.sqrt(5)) / 10 * 30

    X_append = 120
    Y_append = 60

    vertices.append([m, 0, n])
    vertices.append([m, 0, -n])
    vertices.append([-m, 0, n])
    vertices.append([-m, 0, -n])
    vertices.append([0, n, m])
    vertices.append([0, -n, m])
    vertices.append([0, n, -m])
    vertices.append([0, -n, -m])
    vertices.append([n, m, 0])
    vertices.append([-n, m, 0])
    vertices.append([n, -m, 0])
    vertices.append([-n, -m, 0])

    for i in vertices:
        i[0] += X_append
        i[1] += Y_append

    # 正二十面体三角形的点序列
    faces = [[6, 4, 8], [9, 4, 6], [6, 3, 9], [6, 1, 3], [6, 8, 1],
             [8, 10, 1], [8, 0, 10], [8, 4, 0], [4, 2, 0], [4, 9, 2],
             [9, 11, 2], [9, 3, 11], [3, 1, 7], [1, 10, 7], [10, 0, 5],
             [0, 2, 5], [2, 11, 5], [3, 7, 11], [5, 11, 7], [10, 5, 7]]

    return faces


if __name__ == "__main__":
    # 读取标识物图像
    imgObject = cv2.imread("camera_parameters/symbol.png")
    colors = [[20, 111, 93], [110, 115, 20], [32, 7, 21], [52, 44, 98], [119, 21, 39], [17, 76, 78], [53, 89, 55],
              [23, 28, 6], [30, 94, 29], [92, 26, 8], [52, 91, 13], [29, 4, 76], [64, 3, 107], [112, 15, 118],
              [29, 22, 13], [77, 11, 5], [7, 31, 73], [4, 13, 92], [110, 100, 9], [4, 3, 46]]

    index = 0
    # 加载保存的相机矩阵和畸变系数
    npz_file = np.load('biaoding/B.npz')
    mtx = npz_file['mtx']
    dist = npz_file['dist']
    vertices = []
    faces = get_axis()
    # print(vertices)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((24 * 13, 3), np.float32)
    objp[:, :2] = np.mgrid[0:24, 0:13].T.reshape(-1, 2)
    # 定义一个正方形
    axis_qu = np.float32([[0, 0, 0], [0, 20, 0], [20, 20, 0], [20, 0, 0],
                       [0, 0, -20], [0, 20, -20], [20, 20, -20], [20, 0, -20]])

    # 处理camera_parameters/data中的所有图片并输出结果到camera_parameters/out
    images = glob.glob('camera_parameters/data/*.jpg')
    for index in range(0, 27):
        fname = "camera_parameters/data/IMG" + str(index) + ".jpg"
        main(fname, index)
        index += 1
