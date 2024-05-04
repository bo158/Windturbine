import pandas as pd
import numpy as np
import cv2 as cv
#讀取csv文件並將追蹤點對應到影片的第一幀、最後一幀上，並按照距離圓心距離排序
# 读取 CSV 文件并选择第一帧的数据
df = pd.read_csv('D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/output_csv/nom_17_2.csv', delimiter=',')
cap = cv.VideoCapture("D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/test_video/4_21/20240420_17.mp4")

# 圆心坐标
circle_point = (845.43, 499.446)

df_first_frame = df[df['Frame'] == 1].copy()
# 计算每个点到圆心的距离，并添加到 DataFrame 中
df_first_frame.loc[:, 'Distance to Circle Center'] = np.sqrt((df_first_frame['X Coordinate'] - circle_point[0])**2 + (df_first_frame['Y Coordinate'] - circle_point[1])**2)
# 按照距离排序
df_sorted = df_first_frame.sort_values(by='Distance to Circle Center')
# 添加每个点是第几个点的说明
df_sorted['Point Number'] = range(1, len(df_sorted) + 1)

# 显示结果
print(df_sorted)
print(df_sorted['Point'].tolist())
last_frame_number = df['Frame'].iloc[-1]
df_last_frame = df[df['Frame'] == last_frame_number].copy()


ret,frame = cap.read()
ret,frame = cap.read()

total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

# 设置视频的当前帧为最后一帧
cap.set(cv.CAP_PROP_POS_FRAMES, total_frames - 1)

# 读取最后一帧
ret, last_frame = cap.read()

# 绘制每个点的位置并标注它们是第几个点
for index, row in df_sorted.iterrows():
    cv.circle(frame, (int(row['X Coordinate']), int(row['Y Coordinate'])), 3, (0, 255, 0), -1)
    cv.putText(frame, str(row['Point']), (int(row['X Coordinate']), int(row['Y Coordinate'])), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

for index, row in df_last_frame.iterrows():
    cv.circle(last_frame, (int(row['X Coordinate']), int(row['Y Coordinate'])), 3, (0, 255, 0), -1)
    cv.putText(last_frame, str(row['Point']), (int(row['X Coordinate']), int(row['Y Coordinate'])), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)



# 获取图像尺寸
height1, width1, _ = frame.shape
height2, width2, _ = last_frame.shape

# 确定缩小后的尺寸
new_height = max(height1, height2) // 2
new_width = (width1 + width2) // 5

# 将两个图像分别缩小一半
resized_frame = cv.resize(frame, (new_width, new_height))
resized_last_frame = cv.resize(last_frame, (new_width, new_height))


image = np.concatenate((resized_frame, resized_last_frame),axis=1)
# 显示合并后的图像
cv.imshow('Combined Image',image)

while True:
    key = cv.waitKey(0)
    #按esc鍵跳出
    if key == 27:  
        break
cv.destroyAllWindows()
cap.release()

