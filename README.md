# 为压缩空间 删除了大部分图片 仅留网页端测试用图(final_ex/camera_parameters/test)和少量展示用图片



## 环境配置

python == 3.9

opencv-python == 4.4.0.46

matplotilb == 3.8.3



## 运行说明

### 网页版

运行API.py文件后访问http://127.0.0.1:5000

### 本地版

运行deal_and_saze.py将读取camera_parameters/data目录下的原始图像进行处理,然后将生成的AR图像存到camera_parameters/out

### <u>详情见大作业文档</u>



## 文件说明

#### ex1目录

实验1代码, 其中source存放图片资源,KeyPointsMatches.py为实验代码

#### ex2目录

实验2代码, 其中data/cali存放用于标定的棋盘格图片,data/pose存放用于位姿估计和绘图的棋盘格图片,ex2.py为实验代码

## <u>final_ex目录(大作业代码文件)</u>

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