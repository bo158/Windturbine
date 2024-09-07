import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

# 创建 Tkinter 主窗口
root = tk.Tk()
root.withdraw()  # 隐藏主窗口

# 选择输入视频文件
video_path = filedialog.askopenfilename(
    title="選擇需要去畸變影片", 
    filetypes=[("影片文件", "*.MP4;*.avi")]
)

if not video_path:
    print("未選擇影片")
    exit()
else:
    print(f"輸入影片: {video_path}")

# 选择输出文件的保存路径
output_path = filedialog.asksaveasfilename(
    title="選擇保存影片的位置", 
    defaultextension=".mp4", 
    filetypes=[("MP4 文件", "*.mp4"), ("所有文件", "*.*")]
)

if not output_path:
    print("未選擇保存路徑")
    exit()

# 替换成之前计算出的相机参数
mtx = np.array([[3968.50916, 0, 1426.28698],
                [0, 3956.09872, 775.848051],
                [0, 0, 1]])
dist = np.array([0.365264321 ,-3.78425849 , 0.0183564697 , 0.0154046888, 18.6617697])

# 打开视频文件
cap = cv.VideoCapture(video_path)

# 获取视频的属性
fps = cap.get(cv.CAP_PROP_FPS)
frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

# 计算新的相机矩阵
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (frame_width, frame_height), 1, (frame_width, frame_height))

# 裁剪区域
x, y, w, h = roi

# 更新视频写入对象的尺寸为裁剪后的尺寸
cropped_width = w
cropped_height = h

# 创建视频写入对象（使用 MP4V 编码器）
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_path, fourcc, fps, (cropped_width, cropped_height))

# 获取视频总帧数以初始化进度条
total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# 使用 tqdm 显示进度条
for _ in tqdm(range(total_frames), desc="Processing Frames", unit="frame"):
    ret, frame = cap.read()
    if not ret:
        break
    
    # 去畸变
    dst = cv.undistort(frame, mtx, dist, None, newcameramtx)
    
    # 裁剪图像
    dst = dst[y:y+h, x:x+w]
    
 

    
    # 检查是否按下 'q' 键来退出
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
    # 写入视频
    out.write(dst)

# 释放资源
cap.release()
out.release()
cv.destroyAllWindows()

print(f"處理完成，輸出影片已保存至: {output_path}")
