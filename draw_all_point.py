import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.fft import fft
from scipy.signal import find_peaks
#提取多個csv檔案，並使用分別的指定點繪製圖表
# 定义一个函数来计算圆心与点之间的距离
def distance_to_circle(params, x, y):
    xc, yc, r = params
    return np.sqrt((x - xc)**2 + (y - yc)**2) - r

# 定义拟合圆的初始参数
def initial_guess(x, y):
    xc = np.mean(x)
    yc = np.mean(y)
    r = np.mean(np.sqrt((x - xc)**2 + (y - yc)**2))
    return [xc, yc, r]

# 定义文件名和要选择的点的列表
files_and_points = [
    {'file': 'nom_17_2.csv', 'points': [1,11]},
    {'file': '2_17_2.csv', 'points': [1,16]},
]
colors = plt.cm.tab10(np.linspace(0, 1, 50))

# 用于存储每个CSV文件的圆心坐标
average_centers = []

# 绘制子图
fig, axes = plt.subplots(nrows=len(files_and_points), ncols=3, figsize=(16, 9))

for idx, item in enumerate(files_and_points):
    # 读取 CSV 文件
    df = pd.read_csv(f'D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/output_csv/{item["file"]}', delimiter=',')
    
    # 用于存储当前CSV文件中的圆心坐标
    centers = []
    
    for point in item['points']:
        # 提取点的 x 和 y 坐标
        x_values = df[df['Point'] == point]['X Coordinate']
        y_values = df[df['Point'] == point]['Y Coordinate']
        
        # 绘制点
        axes[idx, 0].scatter(x_values, y_values, label=f'Point {point}', color=colors[point-1], s=1)
        
        # 利用最小二乘法拟合圆形
        initial = initial_guess(x_values, y_values)
        result = least_squares(distance_to_circle, initial, args=(x_values, y_values))
        xc, yc, r = result.x
        
        # 添加当前CSV文件中的圆心坐标到列表中
        centers.append((xc, yc))
        
        # 绘制拟合的圆形
        theta = np.linspace(0, 2*np.pi, 100)
        x_fit = xc + r * np.cos(theta)
        y_fit = yc + r * np.sin(theta)
        axes[idx, 0].plot(x_fit, y_fit, color=colors[point-1], linestyle='--')
        axes[idx, 0].legend()
        # 打印圆心坐标和半径
        print(f'For {item["file"]} - Point {point}, Circle Center: ({xc:.2f}, {yc:.2f}), Radius: {r:.2f}')

    # 计算当前CSV文件的平均圆心坐标并添加到列表中
    average_center = np.mean(centers, axis=0)
    average_centers.append(average_center)
    print(average_center)
    print(average_centers)
    #列印平均圓心
 

#x,y座標隨時間變化圖
    frame_rate = 120  # 假设帧率为120帧/秒
    frames_per_millisecond = frame_rate / 1000  # 每毫秒的帧数
    
    frame_numbers = df[df['Point'] == item['points'][0]]['Frame']
    times_in_milliseconds = frame_numbers / frames_per_millisecond
    filename_without_extension = item["file"].split(".")[0]
    for point in item['points']:
       
        x_values = df[df['Point'] == point]['X Coordinate']
        axes[idx, 1].scatter(times_in_milliseconds, x_values, label=f'X Coordinate of Point {point}', s=1)
        axes[idx, 1].set_title(f'{filename_without_extension}-X Coordinate of Point')
        axes[idx, 1].set_xlabel('Time (ms)')
        axes[idx, 1].set_ylabel('X Coordinate')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True)

        y_values = df[df['Point'] == point]['Y Coordinate']
        axes[idx, 2].scatter(times_in_milliseconds, y_values, label=f'Y Coordinate of Point {point}', s=1)
        axes[idx, 2].set_title(f'{filename_without_extension}-Y Coordinate of Point')
        axes[idx, 2].set_xlabel('Time (ms)')
        axes[idx, 2].set_ylabel('Y Coordinate')
        axes[idx, 2].legend()
        axes[idx, 2].grid(True)

plt.tight_layout()
plt.show()


# 纵轴标准化函数
def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)
def standardize(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    return (data - mean_val) / std_val
#x,y座標隨時間變化圖(標準化)
fig, axes = plt.subplots(nrows=len(files_and_points), ncols=2, figsize=(16, 9))
for idx, item in enumerate(files_and_points):
    # 读取 CSV 文件
    df = pd.read_csv(f'D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/output_csv/{item["file"]}', delimiter=',')

    frames_per_millisecond = frame_rate / 1000  # 每毫秒的帧数
    
    frame_numbers = df[df['Point'] == item['points'][0]]['Frame']
    times_in_milliseconds = frame_numbers / frames_per_millisecond
    filename_without_extension = item["file"].split(".")[0]
    for point in item['points']:
        x_values = df[df['Point'] == point]['X Coordinate']
       
        axes[idx, 0].scatter(times_in_milliseconds, normalize(x_values), label=f'X Coordinate of Point {point}', s=1)
        axes[idx, 0].set_title(f'{filename_without_extension}-X Coordinate of Point')
        axes[idx, 0].set_xlabel('Time (ms)')
        axes[idx, 0].set_ylabel('X Coordinate')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True)

        y_values = df[df['Point'] == point]['Y Coordinate']
        axes[idx, 1].scatter(times_in_milliseconds,normalize( y_values), label=f'Y Coordinate of Point {point}', s=1)
        axes[idx, 1].set_title(f'{filename_without_extension}-Y Coordinate of Point')
        axes[idx, 1].set_xlabel('Time (ms)')
        axes[idx, 1].set_ylabel('Y Coordinate')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True)

plt.tight_layout()
plt.show()

#x,y座標隨時間變化圖傅立葉轉換
fig, axes = plt.subplots(nrows=len(files_and_points), ncols=2, figsize=(16, 9))
for idx, item in enumerate(files_and_points):
    # 读取 CSV 文件
    df = pd.read_csv(f'D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/output_csv/{item["file"]}', delimiter=',')
    fs = frame_rate
    T = 1 / fs
    filename_without_extension = item["file"].split(".")[0]
    for point in item['points']:
        x_values = df[df['Point'] == point]['X Coordinate']
        N = len(x_values)
        xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
        f_x_values = fft(x_values)
       
        axes[idx, 0].plot(xf,2.0 / N * np.abs(f_x_values[:N//2]), label=f'X Coordinate of Point {point}')
        axes[idx, 0].set_title(f'{filename_without_extension}-X Coordinate of Point')
        axes[idx, 0].set_xlabel('Frequency (Hz)')
        axes[idx, 0].set_ylabel('Amplitude')
        axes[idx, 0].legend()
        axes[idx, 0].grid(True)
        peaks, _ = find_peaks(2.0 / N * np.abs(f_x_values[:N//2]), 10)
        axes[idx, 0].plot(xf[peaks], 2.0 / N * np.abs(f_x_values[:N//2])[peaks], "x", label='_nolegend_', color='red')
        for peak in peaks:
            axes[idx, 0].annotate(f'{xf[peak]:.2f} Hz', xy=(xf[peak], 0), xytext=(xf[peak], 0.1))


        y_values = df[df['Point'] == point]['Y Coordinate']
        f_y_values = fft(y_values)
        axes[idx, 1].plot(xf,2.0 / N * np.abs(f_y_values[:N//2]), label=f'Y Coordinate of Point {point}')
        axes[idx, 1].set_title(f'{filename_without_extension}-Y Coordinate of Point')
        axes[idx, 1].set_xlabel('Frequency (Hz)')
        axes[idx, 1].set_ylabel('Amplitude')
        axes[idx, 1].legend()
        axes[idx, 1].grid(True)
        peaks, _ = find_peaks(2.0 / N * np.abs(f_y_values[:N//2]),10)
        axes[idx, 1].plot(xf[peaks], 2.0 / N * np.abs(f_y_values[:N//2])[peaks], "x", label='_nolegend_', color='red')
        for peak in peaks:
            axes[idx, 1].annotate(f'{xf[peak]:.2f} Hz', xy=(xf[peak], 0), xytext=(xf[peak], 0.1))

plt.tight_layout()
plt.show()

# 角速度计算函数 calculate_angular_velocity
def calculate_angular_velocity(x_values, y_values, xc, yc, frame_rate):
    # 计算每一帧的时间间隔
    dt = 1 / frame_rate
    
    # 计算位置相对于圆心的偏移量
    dx = x_values - xc
    dy = y_values - yc
    # 计算向量角度
    theta=np.arctan2(dy,dx)
    theta_series = pd.Series(theta)
# 使用 reset_index() 方法重置索引
    df = theta_series.reset_index(drop=True)
    for i in range(1, len(df)-1):
        diff = df[i] - df[i-1]
        if diff > np.pi:
            df[i:] -= 2 * np.pi
        elif diff < -np.pi:
            df[i:] += 2 * np.pi
    # 计算角速度
    angular_def = np.gradient(df)
    angular_velocity = angular_def/dt
    return angular_velocity
# 角速度隨時間變化圖
fig, axes = plt.subplots(nrows=len(files_and_points), ncols=1, figsize=(16, 9))

for idx, item in enumerate(files_and_points):
    # 读取 CSV 文件
    df = pd.read_csv(f'D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/output_csv/{item["file"]}', delimiter=',')
    filename_without_extension = item["file"].split(".")[0]
    all_angular_velocities = []
    for point in item['points']:
        x_values = df[df['Point'] == point]['X Coordinate']
        y_values = df[df['Point'] == point]['Y Coordinate']
        # 计算角速度
        angular_velocity = calculate_angular_velocity(x_values, y_values, average_centers[idx][0], average_centers[idx][1], frame_rate)
        all_angular_velocities.append(angular_velocity)
        # 绘制角速度随时间的变化图
        frame_numbers = df[df['Point'] == point]['Frame']
        times_in_milliseconds = frame_numbers / (frame_rate / 1000)  # 将帧转换为毫秒
       
        axes[idx].plot(times_in_milliseconds, angular_velocity, label=f'Angular Velocity of Point {point}')
        axes[idx].set_title(f'{filename_without_extension}-Angular Velocity of Point')
        axes[idx].set_xlabel('Time (ms)')
        axes[idx].set_ylabel('Angular Velocity (rad/s)')
        axes[idx].legend()
        axes[idx].grid(True)
    average_angular_velocity = np.mean(all_angular_velocities, axis=0)
    total_average_angular_velocity = np.mean(average_angular_velocity)
    axes[idx].axhline(y=total_average_angular_velocity, color='green', linestyle='-.', label=f'Total Average Angular Velocity: {total_average_angular_velocity:.2f} rad/s')
    axes[idx].legend()

plt.tight_layout()
plt.show()

def calculate_edgewise_deplacement(x_values1, y_values1,x_values2,y_values2, xc, yc):
    # 计算每一帧的时间间隔
    
    dx1 = x_values1 - xc
    dy1 = y_values1 - yc
    dx2 = x_values2 - x_values1
    dy2 = y_values2 - y_values1
    
    r_x= x_values2 - x_values1
    r_y= y_values2 - y_values1
    r=np.sqrt(r_x**2+r_y**2)
    # 计算向量角度
    theta1=np.arctan2(dy1,dx1)
    theta2=np.arctan2(dy2,dx2)
    theta1_series = pd.Series(theta1)
    theta2_series = pd.Series(theta2)
# 使用 reset_index() 方法重置索引
    df1 = theta1_series.reset_index(drop=True)
    df2 = theta2_series.reset_index(drop=True)
    for i in range(1, len(df1)-1):
        diff1 = df1[i] - df1[i-1]
        if diff1 > np.pi:
            df1[i:] -= 2 * np.pi
        elif diff1 < -np.pi:
            df1[i:] += 2 * np.pi
    for i in range(1, len(df2)-1):
        diff2 = df2[i] - df2[i-1]
        if diff2 > np.pi:
            df2[i:] -= 2 * np.pi
        elif diff2 < -np.pi:
            df2[i:] += 2 * np.pi
    # 计算角速度
    angular_def=np.gradient(df2-df1)
    edgewise_deplacement =r*angular_def
    return edgewise_deplacement
# 绘制子图
#edgewise方向位移圖
fig, axes = plt.subplots(nrows=len(files_and_points), ncols=1, figsize=(16, 9))

for idx, item in enumerate(files_and_points):
    # 读取 CSV 文件
    df = pd.read_csv(f'D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/output_csv/{item["file"]}', delimiter=',')
    filename_without_extension = item["file"].split(".")[0]
    first_point = item['points'][0]
    x_values1 = df[df['Point'] == first_point]['X Coordinate']
    y_values1 = df[df['Point'] == first_point]['Y Coordinate']
    for point in item['points'][1:]:
        x_values2 = df[df['Point'] == point]['X Coordinate']
        y_values2= df[df['Point'] == point]['Y Coordinate']
        x_values_series1 = pd.Series(x_values1)
        y_values_series1 = pd.Series(y_values1)
        x_values_series2 = pd.Series(x_values2)
        y_values_series2 = pd.Series(y_values2)
        x_values1 = x_values_series1.reset_index(drop=True)
        y_values1 = y_values_series1.reset_index(drop=True)
        x_values2 = x_values_series2.reset_index(drop=True)
        y_values2 = y_values_series2.reset_index(drop=True)
        # 计算角速度
        edgewise_deplacement = calculate_edgewise_deplacement(x_values1, y_values1,x_values2,y_values2, average_centers[idx][0], average_centers[idx][1])
        # 绘制角速度随时间的变化图
        frame_numbers = df[df['Point'] == point]['Frame']
        times_in_milliseconds = frame_numbers / (frame_rate / 1000)  # 将帧转换为毫秒
        
        axes[idx].plot(times_in_milliseconds, edgewise_deplacement, label=f'edgewisedeplacement of Point {point}')
        axes[idx].set_title(f'{filename_without_extension}-edgewise deplacement of Point')
        axes[idx].set_xlabel('Time (ms)')
        axes[idx].set_ylabel('edgewise deplacement(pixel)')
        axes[idx].legend()
        axes[idx].grid(True)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(nrows=len(files_and_points), ncols=1, figsize=(16, 9))

for idx, item in enumerate(files_and_points):
    # 读取 CSV 文件
    df = pd.read_csv(f'D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/output_csv/{item["file"]}', delimiter=',')
    filename_without_extension = item["file"].split(".")[0]
    fs = frame_rate
    T = 1 / fs
    first_point = item['points'][0]
    x_values1 = df[df['Point'] == first_point]['X Coordinate']
    y_values1 = df[df['Point'] == first_point]['Y Coordinate']
    for point in item['points'][1:]:
        x_values2 = df[df['Point'] == point]['X Coordinate']
        y_values2= df[df['Point'] == point]['Y Coordinate']
        x_values_series1 = pd.Series(x_values1)
        y_values_series1 = pd.Series(y_values1)
        x_values_series2 = pd.Series(x_values2)
        y_values_series2 = pd.Series(y_values2)
        x_values1 = x_values_series1.reset_index(drop=True)
        y_values1 = y_values_series1.reset_index(drop=True)
        x_values2 = x_values_series2.reset_index(drop=True)
        y_values2 = y_values_series2.reset_index(drop=True)
        # 计算角速度
        edgewise_deplacement = calculate_edgewise_deplacement(x_values1, y_values1,x_values2,y_values2, average_centers[idx][0], average_centers[idx][1])
        # 绘制角速度随时间的变化图
        N = len(edgewise_deplacement)
        xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
        f_edgewise_deplacement = fft(edgewise_deplacement)
      
        axes[idx].plot(xf,2.0 / N * np.abs(f_edgewise_deplacement[:N//2]), label=f'edgewisedeplacement of Point {point}')
        axes[idx].set_title(f'{filename_without_extension}-edgewise deplacement of Point')
        axes[idx].set_xlabel('Frequency (Hz)')
        axes[idx].set_ylabel('Amplitude')
        axes[idx].legend()
        axes[idx].grid(True)
        peaks, _ = find_peaks(2.0 / N * np.abs(f_edgewise_deplacement[:N//2]),0.03)
        axes[idx].plot(xf[peaks], 2.0 / N * np.abs(f_edgewise_deplacement[:N//2])[peaks], "x", label='_nolegend_', color='red')
        for peak in peaks:
            axes[idx].annotate(f'{xf[peak]:.2f} Hz', xy=(xf[peak], 0), xytext=(xf[peak], 0.1))

plt.tight_layout()
plt.show()

def calculate_flapwise_deplacement(x_values1, y_values1,x_values2,y_values2):
    # 计算每一帧的时间间隔    
    r_x= x_values2 - x_values1
    r_y= y_values2 - y_values1
    r=np.sqrt(r_x**2+r_y**2)
    r_def=r-r[0]
    return r_def
# 绘制子图
fig, axes = plt.subplots(nrows=len(files_and_points), ncols=1, figsize=(16, 9))

for idx, item in enumerate(files_and_points):
    # 读取 CSV 文件
    df = pd.read_csv(f'D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/output_csv/{item["file"]}', delimiter=',')
    filename_without_extension = item["file"].split(".")[0]
    for point in item['points'][1:8:2]:
        x_values1 = df[df['Point'] == point]['X Coordinate']
        y_values1 = df[df['Point'] == point]['Y Coordinate']
        x_values2 = df[df['Point'] == point+1]['X Coordinate']
        y_values2 = df[df['Point'] == point+1]['Y Coordinate']
        x_values_series1 = pd.Series(x_values1)
        y_values_series1 = pd.Series(y_values1)
        x_values_series2 = pd.Series(x_values2)
        y_values_series2 = pd.Series(y_values2)
        x_values1 = x_values_series1.reset_index(drop=True)
        y_values1 = y_values_series1.reset_index(drop=True)
        x_values2 = x_values_series2.reset_index(drop=True)
        y_values2 = y_values_series2.reset_index(drop=True)
        # 计算角速度
        flapwise_deplacement = calculate_flapwise_deplacement(x_values1, y_values1,x_values2,y_values2)
        # 绘制角速度随时间的变化图
        frame_numbers = df[df['Point'] == point]['Frame']
        times_in_milliseconds = frame_numbers / (frame_rate / 1000)  # 将帧转换为毫秒
     
        axes[idx].plot(times_in_milliseconds, flapwise_deplacement, label=f'flapwise_deplacement of Point {point}')
        axes[idx].set_title(f'{filename_without_extension}-flapwise deplacement of Point')
        axes[idx].set_xlabel('Time (ms)')
        axes[idx].set_ylabel('flapwise deplacement(pixel)')
        axes[idx].legend()
        axes[idx].grid(True)
plt.tight_layout()
plt.show()

def calculate_elongation(x_values, y_values, xc, yc):
    dx = x_values - xc
    dy = y_values - yc
    r=np.sqrt(dx**2+dy**2)
    r_df=r-r[0]
    return r_df
# 绘制子图
fig, axes = plt.subplots(nrows=len(files_and_points), ncols=1, figsize=(16, 9))

for idx, item in enumerate(files_and_points):
    # 读取 CSV 文件
    df = pd.read_csv(f'D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/output_csv/{item["file"]}', delimiter=',')
    filename_without_extension = item["file"].split(".")[0]
    all_angular_velocities = []
    for point in item['points']:
        x_values = df[df['Point'] == point]['X Coordinate']
        y_values = df[df['Point'] == point]['Y Coordinate']
        x_values_series = pd.Series(x_values)
        y_values_series = pd.Series(y_values)
        x_values = x_values_series.reset_index(drop=True)
        y_values = y_values_series.reset_index(drop=True)
        # 计算角速度
        elongation = calculate_elongation(x_values, y_values, average_centers[idx][0], average_centers[idx][1])
        # 绘制角速度随时间的变化图
        frame_numbers = df[df['Point'] == point]['Frame']
        times_in_milliseconds = frame_numbers / (frame_rate / 1000)  # 将帧转换为毫秒
        
        axes[idx].plot(times_in_milliseconds,elongation, label=f'elongation of Point {point}')
        axes[idx].set_title(f'{filename_without_extension}-elongation of Point')
        axes[idx].set_xlabel('Time (ms)')
        axes[idx].set_ylabel('elongation(pixel)')
        axes[idx].legend()
        axes[idx].grid(True)
plt.tight_layout()
plt.show()