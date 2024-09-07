import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import filedialog
import os

# 终止条件
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 准备对象点，例如 (0,0,0), (30,0,0), (60,0,0) ....,(180,150,0)
objp = np.zeros((6*9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 24  # 假设每个方格大小为 30mm

# 存储所有图像的对象点和图像点
objpoints = []  # 3D点在实际世界中的坐标
imgpoints = []  # 2D点在图像平面中的坐标

# 创建 Tkinter 对话框以选择视频文件
root = tk.Tk()
root.withdraw()  # 隐藏主窗口
video_path = filedialog.askopenfilename(title="选择棋盘格视频", filetypes=[("视频文件", "*.mp4;*.avi")])

if not video_path:
    print("未选择视频文件")
    exit()

# 打开视频文件
cap = cv.VideoCapture(video_path)

frame_count = 0
selected_frames = 0
frame_interval = 50  # 每隔多少帧选择一帧

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_interval == 0:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (9, 6), None)
        
        if ret:
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)
            selected_frames += 1
            
            # 绘制并显示角点
            cv.drawChessboardCorners(frame, (9, 6), corners2, ret)
            cv.imshow('Selected Frame', frame)
            cv.waitKey(500)

cap.release()
cv.destroyAllWindows()

# 打印数量以调试
print("对象点数量:", len(objpoints))
print("图像点数量:", len(imgpoints))

if len(objpoints) == 0 or len(objpoints) != len(imgpoints):
    print("没有足够的角点或角点数量不匹配")
    exit()

# 标定相机
ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# 打印相机参数
print("相机矩阵：")
print(mtx)
print("畸变系数：")
print(dist)

