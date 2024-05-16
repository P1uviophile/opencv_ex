## 环境配置

python == 3.9

opencv-python == 4.4.0.46

matplotilb == 3.8.3



## 运行说明

### 运行前提示

若想用自己相机拍摄请将biaoding/cail文件夹下的图片替换成自己相机拍摄的棋盘格图片后运行biaoding/calibrate.py; 同时请将camera_parameters下的data和test下的图片替换成自己相机拍摄的图片; 替换标识物图片请在理解下文的项目流程的前提下自行更换标识物与相关参数

### 网页版

运行API.py文件后访问http://127.0.0.1:5000

### 本地版

运行deal_and_saze.py将读取camera_parameters/data目录下的原始图像进行处理,然后将生成的AR图像存到camera_parameters/out

### <u>详情见下文</u>



## 文件说明

### biaoding目录

cali目录存放用于相机标定的棋盘格图片,calibrate.py完成相机标定并保存得到的内参到B.npz文件中

### camera_parameters目录

data存放用于处理的原始图像

out存放绘制AR几何体的图像

test中的图像用于测试网页端的在线处理功能

symbol.png是标识物的原始图像

test.py文件用于重命名data中的文件

生成颜色数组.py可以不用管

### static目录

用于存放网页端的静态资源

camera_parameters目录包含上面的同名目录的data和out内容,用于网页端展示AR图像

tmp用于临时存放在线生成的AR图像

### templates目录

存放html文件

### API.py

实现网页端的py文件,运行后访问http://127.0.0.1:5000即可使用网页版

### API_process.py

为API.py提供平面标志物定位及AR展示服务

### deal_and_saze.py

本地版API_process.py,读取camera_parameters/data目录下的原始图像进行处理后将生成的AR图像存到camera_parameters/out





***\*项目流程\**** 

**1.** ***\*平面标志物定位\****

**1.1** ***\*相机标定\****

![wps1](https://github.com/P1uviophile/opencv_ex/assets/95516646/62fed83a-3013-4736-be61-3c6e8e766fb4)

calibrate.py使用实验二中的相机标定方法,读取cail文件夹中的棋盘格图片用于棋盘格标定相机内参,最后将标定的相机内参保存



------------------

\# 函数返回 整体RMS重投影误差, 相机矩阵, 畸变系数, 旋转和平移向量等
	ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

np.savez('B.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

----



通过np.saze函数保存相机内参到B.npz中, 获取标识物位姿信息时从中读取内参使用.

 

**2.** ***\*获取标识物位姿信息\****

![wps2](https://github.com/P1uviophile/opencv_ex/assets/95516646/d6659b6d-f34d-4d64-b6ae-a3761cee1b5f)

标识物图片和将要处理的图片都存储在该目录下的data文件夹中

![wps3](https://github.com/P1uviophile/opencv_ex/assets/95516646/a4fa58b0-2345-43ca-afcf-697e0daa8ae3)

两个py文件都实现了获取标识物位姿信息和绘制几何体,不过API_process用于给

网页API提供服务,函数run返回处理后的图片; deal_and_saze用于本地处理图片,主函	数保存处理后的图片到上面目录中的out文件夹

 

**2.1** ***\*标志物定位\****

使用SIFT方法提取对应点后使用FLANN方法处理SIFT特征点，获得匹配点,再使用findHomograhy方法计算单应矩阵H,再通过映射点群的方法获得标识物的四个角点:
![wps4](https://github.com/P1uviophile/opencv_ex/assets/95516646/973caba5-ec93-431c-b8df-8e94b6f668fa)

**2.2** ***\*位姿计算\****

![wps5](https://github.com/P1uviophile/opencv_ex/assets/95516646/045a0e36-3c32-44ff-88d6-2114eef8378a)

通过标识物的四个角点和之前保存的相机内参数和参考标识物的四个角点(因为在同一平面所以z轴全为0,xy长度需要测量),调用solvePnP方法(EPNP算法)计算标识物的位姿信息,得到rvec和tvec

**3.** ***\*投影几何体坐标并绘制\****

事先通过数学方法计算得到二十面体的坐标信息,根据实际情况放缩坐标以适应标识物大小:

![wps6](https://github.com/P1uviophile/opencv_ex/assets/95516646/3fbbddd4-1061-4767-aaa3-7e5f37762db8)

 

顺便再绘制一个以左上角为顶点的正方体检测投影偏差:
![wps7](https://github.com/P1uviophile/opencv_ex/assets/95516646/a153a18d-b3e3-480f-b72b-238e4a7b1638)

![wps8](https://github.com/P1uviophile/opencv_ex/assets/95516646/4cc46c97-a4c0-4baf-8646-82072393f1a1)

 

调用project方法将三维坐标投影到图像上,得到画图用的二维坐标信息:

![wps9](https://github.com/P1uviophile/opencv_ex/assets/95516646/c600f1e4-9189-4364-b5aa-ef4104f9af5c)

根据计算得到的坐标信息和面信息绘制二十面体:
![wps10](https://github.com/P1uviophile/opencv_ex/assets/95516646/3064323b-8a2e-4fb6-bdb0-5e2a4bda0e0c)

 

 

**4.** ***\*网页API实现\****

 

![wps11](https://github.com/P1uviophile/opencv_ex/assets/95516646/85d1814d-cfb0-4edc-b213-d72d5af65b68)

 

提供在线检测图片标识物并绘制模型和展示21幅不同角度和远近的AR图像

 

**5.** ***\*样例展示\****

 

![wps12](https://github.com/P1uviophile/opencv_ex/assets/95516646/2a963e4b-2870-4122-9640-241d6234ac63)

![wps13](https://github.com/P1uviophile/opencv_ex/assets/95516646/9a0aa791-4770-4d19-b710-bc47978d100c)

![wps14](https://github.com/P1uviophile/opencv_ex/assets/95516646/98dec6c1-cb6b-4922-8e81-0cd08aeaa583)
