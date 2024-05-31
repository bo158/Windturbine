import pandas as pd
import numpy as np
import cv2 as cv
from scipy.optimize import least_squares
import random
import csv
# 指定数据类型
dtype_spec = {
    'X Coordinate': float,
    'Y Coordinate': float,
    'Frame': int,
    'ROI number': int,
    'Point': int,
    # 在这里添加第 15 列和第 16 列的实际列名
    'column_15_name': str,
    'column_16_name': str
}
csv_filename ='D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/output_csv/5_26/1_40_2.csv'
# 读取 CSV 文件
df = pd.read_csv(csv_filename, delimiter=',', encoding='utf-8', dtype=dtype_spec)
cap = cv.VideoCapture("D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/test_video/5_26/IMG_1892.MOV")
max_frame = df['Frame'].max()
ret, frame = cap.read()
# 设置视频的当前帧为最后一帧
cap.set(cv.CAP_PROP_POS_FRAMES, max_frame - 1)
# 读取最后一帧
ret, last_frame = cap.read()
def initial_guess(x, y):
    xc = np.mean(x)
    yc = np.mean(y)
    r = np.mean(np.sqrt((x - xc)**2 + (y - yc)**2))
    return [xc, yc, r]

def distance_to_circle(params, x, y):
    xc, yc, r = params
    return np.sqrt((x - xc)**2 + (y - yc)**2) - r

df['ROI number'] = df['ROI number'].astype(int)
df['Point'] = df['Point'].astype(int)
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
offset_range = 20
index=0
for roi_index, roi_number in enumerate(sorted(df['ROI number'].unique())):
    roi_points = df[df['ROI number'] == roi_number]
    centers = []
    errors = []
    color = colors[roi_index % len(colors)]  # 根据 ROI number 的索引选择颜色
    # 在当前 ROI number 下遍历所有的 Point
    for point in sorted(roi_points['Point'].unique()):
        index=index+1
        point_data = roi_points[roi_points['Point'] == point]
        x_values = point_data['X Coordinate']
        y_values = point_data['Y Coordinate']
        x_values_series = pd.Series(x_values)
        y_values_series = pd.Series(y_values)
        x_values = x_values_series.reset_index(drop=True)
        y_values = y_values_series.reset_index(drop=True)
        initial = initial_guess(x_values, y_values)
        result = least_squares(distance_to_circle, initial, args=(x_values, y_values))
        xc, yc, r = result.x
        centers.append((xc, yc))
        residuals = np.sqrt((x_values - xc)**2 + (y_values - yc)**2) - r
        fit_error = abs(np.mean(residuals))   
        errors.append(fit_error) 
        with open(csv_filename, mode='r', newline='',encoding='utf-8') as file:
            reader = csv.reader(file)
            rows = list(reader)
            rows[index][8:10] =[xc,yc]
        with open(csv_filename, mode='w', newline='',encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

        offset_x = random.randint(-offset_range, offset_range)
        offset_y = random.randint(-offset_range, offset_range)

        cv.circle(frame, (int(x_values[0]), int(y_values[0])), 4, color, -1)
        cv.putText(frame, str(point), (int(x_values[0]+offset_x), int(y_values[0]+offset_y)), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv.circle(last_frame, (int(x_values.iloc[-1]), int(y_values.iloc[-1])), 4, color, -1)
        cv.putText(last_frame, str(point), (int(x_values.iloc[-1]+offset_x), int(y_values.iloc[-1]+offset_y)), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    centers_array = np.array(centers)
    median_center = np.median(centers_array, axis=0)
    print(median_center)
    mad = np.median(np.abs(centers_array - median_center), axis=0)

    # 使用 MAD 设定阈值，假设使用 5 倍的 MAD
    threshold = 5 * mad
    selected_centers = [center for center in centers if all(abs(center - median_center) <= threshold)]
    selected_centers1 = np.array([center if all(abs(center - median_center) <= threshold) else (np.nan, np.nan) for center in centers], dtype=object)
    df_first_frame = roi_points[roi_points['Frame'] == 1].copy()
    # 计算每个点到圆心的距离，并添加到 DataFrame 中
    average_center = np.mean(np.array(selected_centers), axis=0)
    print(average_center)
    df_first_frame.loc[:, 'Distance to Circle Center'] = np.sqrt((df_first_frame['X Coordinate'] - average_center[0])**2 + (df_first_frame['Y Coordinate'] - average_center[1])**2)
    df_first_frame.loc[:, 'Selected Center'] = selected_centers1[:, 0]  # 使用第一列作为选定的中心
    # 将中心点信息存储为元组列表
    center_tuples = [center if center is not None else (np.nan, np.nan) for center in centers]
    df_first_frame['Center'] = center_tuples
    
    df_first_frame.loc[:, 'errors'] = errors
    # 按照距离排序
    df_sorted = df_first_frame.sort_values(by='Distance to Circle Center')
    # 添加每个点是第几个点的说明
    df_sorted['Point Number'] = range(1, len(df_sorted) + 1)
    # 显示结果
    columns_of_interest = ['Frame', 'ROI number', 'Point','Radius', 'Distance to Circle Center', 'errors', 'Selected Center', 'Center']
    df_sorted = df_sorted[columns_of_interest]
    print(df_sorted)
# 只显示有被选中中心的点
    selected_points = df_sorted['Point'][df_sorted['Selected Center'].notnull()].tolist()
    print(selected_points)
# 绘制每个点的位置并标注它们是第几个点
    
image = np.concatenate((frame, last_frame), axis=1)
resized_img = cv.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
# 显示合并后的图像
cv.imshow('Combined Image', resized_img)

while True:
    key = cv.waitKey(0)  # 等待 1 毫秒，单位是毫秒
    # 按 ESC 键跳出或者窗口关闭
    if key == 27 or cv.getWindowProperty('Combined Image', cv.WND_PROP_VISIBLE) < 1:  
        break

cv.destroyAllWindows()
cap.release()
