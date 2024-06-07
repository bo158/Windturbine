import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.fft import fft
from scipy.signal import find_peaks
from matplotlib.font_manager import FontProperties
import os
import warnings
from matplotlib.cbook import MatplotlibDeprecationWarning 
warnings.filterwarnings("ignore", category=FutureWarning, module="scipy")
warnings.filterwarnings("ignore", category=MatplotlibDeprecationWarning)

def normalized(data):
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)
def standardized(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    return (data - mean_val) / std_val
def resort(x_values):
    x_values_series = pd.Series(x_values)
    x_values = x_values_series.reset_index(drop=True)
    return x_values
def data_limit(data):
    data=data[:plot_range]
    return data
def calculate_angular_velocity(x_values, y_values, xc, yc, frame_rate):
    dt = 1 / frame_rate
    # 计算位置相对于圆心的偏移量
    dx = x_values - xc
    dy = y_values - yc
    theta=np.arctan2(dy,dx)
    theta_series = pd.Series(theta)
    df = theta_series.reset_index(drop=True)
    for i in range(1, len(df)-1):
        diff = df[i] - df[i-1]
        if diff > np.pi:
            df[i:] -= 2 * np.pi
        elif diff < -np.pi:
            df[i:] += 2 * np.pi
    angular_def = np.gradient(df)
    angular_velocity = -angular_def/dt
    return angular_velocity
def calculate_edgewise_deplacement(x_values1, y_values1,x_values2,y_values2, xc, yc):
    
    dx1 = x_values1 - xc
    dy1 = y_values1 - yc
    dx2 = x_values2 - x_values1
    dy2 = y_values2 - y_values1
    
    r_x= x_values2 - x_values1
    r_y= y_values2 - y_values1

    r=np.sqrt(r_x**2+r_y**2)
    
    # 計算向量角度
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
    # 計算角度梯度
    angular_def=df2-df1
    angular_def=angular_def-np.mean(angular_def)
    edgewise_deplacement =r_el*angular_def
    return edgewise_deplacement
def calculate_flapwise(radius):
    r_df=radius-np.mean(radius)
    return r_df
def calculate_elongation(x_values, y_values, xc, yc):
    dx = x_values - xc
    dy = y_values - yc
    r=np.sqrt(dx**2+dy**2)
    print(np.mean(r))
    r_df=r-np.mean(r)
    return r_df
def data_process(x,y1,y2,data_pross,xlabel,ylabel_1,ylabel_2,unity,fps):
    global pr_type
    pr_type=''
    peaks=0
    if data_pross=='1':
        y1=normalized(y1)
        y2=normalized(y2)
        unity=''
        pr_type='(Normalized)'
    elif data_pross =='2':
        y1=standardized(y1)
        y2=standardized(y2)
        unity=''
        pr_type='(Standardized)'
    elif data_pross =='3':
        T = 1 / fps
        N = len(x)
        x= np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
        y1 =2.0 / N * np.abs(fft(y1)[:N//2]) 
        y2 =2.0 / N * np.abs(fft(y2)[:N//2])
        xlabel='Frequency (Hz)'
        ylabel_1='Amplitude'
        ylabel_2='Amplitude'
        unity=''
        pr_type='(FFT)'
        peaks, _ = find_peaks((y1), 20)
    return x,y1,y2,xlabel,ylabel_1,ylabel_2,unity,peaks
def clac_func(plot_type,x_values1,y_values1,x_values2,y_values2,center_x,center_y,radius,fps):
    global title
    y2=[0,1,2]
    xlabel='Time(ms)'
    ylabel_2=''
    if plot_type=='1':
        y1=x_values2
        y2=y_values2
        ylabel_1='X Coordinate'
        ylabel_2='Y Coordinate'
        unity='(pixel)'
        title='X,Y Scatter Plot'
    elif plot_type=='2':
        y1=x_values2
        y2=y_values2
        ylabel_1='X Coordinate'
        ylabel_2='Y Coordinate'
        unity='(pixel)'
        title='X,Y Coordinate'
    elif plot_type=='3':
        y1=calculate_angular_velocity(x_values2, y_values2, center_x, center_y, fps)
        ylabel_1='Angular Velocity'
        unity='(rad/s)'
        title='Angular Velocity'
    elif plot_type=='4':
        y1=calculate_edgewise_deplacement(x_values1, y_values1,x_values2,y_values2, center_x, center_y)
        ylabel_1='Displacement'
        unity='(pixel)'
        title='Edgewise Direction Displacement'
    elif plot_type=='5':
        y1=calculate_flapwise(radius)
        ylabel_1='Radius'
        unity='(pixel)'
        title='圓點標記的半徑變化量'
    elif plot_type=='6':
        y1=calculate_elongation(x_values2, y_values2, center_x,center_y)
        ylabel_1='Displacement'
        unity='(pixel)'
        title='Longitudinal Direction Displacement'
    return y1,y2,xlabel,ylabel_1,ylabel_2,unity
def lengend_func(lengend,state,blade_index,point_index,video_number,avg,amp,std,unity):
        # 创建一个空的字符串来存储组合结果
    concat = ''
    # 遍历 legend 列表并根据其值组合相应的变量
    for item in lengend:
        if item == 1:
            concat += f'{state}'
        elif item == 2:
            concat += f'  葉片{blade_index}'
        elif item == 3:
            concat += f'  點{point_index}'
        elif item == 4:
            concat += f' 數據{video_number}'
        elif item == 5:
            concat += f'  Avg：{f"{avg:.2f}"}{unity}'
        elif item == 6:
            concat += f'  amplitude：{f"{amp:.2f}"}{unity}'
        elif item == 7:
            concat += f'  std：{f"{std:.2f}"}{unity}'   
    return concat
def group_func(selected_files,selected_blades,selected_points,plot_type,data_pross,group_type,lengend):
    global font_prop,plot_range, r_el
    fig = plt.figure(figsize=(14,7))
    plot_range=3000
    plot_index=-1
    font_path = 'C:/WINDOWS/Fonts/msjh.ttc'  # 根据系统中的字体路径修改
    font_prop = FontProperties(fname=font_path)
    if group_type=='1':#文件分組 
        nrows=len(selected_files)
        for i, file_info in enumerate(selected_files):
            colors = plt.cm.viridis(np.linspace(0, 1, 50))
            plot_index=plot_index+1
            df = pd.read_csv(os.path.join(file_path, file_info['file']), delimiter=',', low_memory=False)
            for j, blade_index in enumerate(selected_blades):
                df_blade = df[df['ROI number'] == blade_index]
                for w,point_index in enumerate(selected_points):
                    point_coordinates = file_info[f'points{blade_index}'][point_index - 1]
                    if point_coordinates=='':
                        pass
                    else:
                        first_point_coordinates = file_info[f'points{blade_index}'][0]
                        x_values1= resort(df_blade[df_blade['Point'] == first_point_coordinates]['X Coordinate'])[:plot_range]
                        y_values1= 1079-resort(df_blade[df_blade['Point'] == first_point_coordinates]['Y Coordinate'])[:plot_range]
                        x_values2= resort(df_blade[df_blade['Point'] == point_coordinates]['X Coordinate'])[:plot_range]
                        y_values2= 1079-resort(df_blade[df_blade['Point'] == point_coordinates]['Y Coordinate'])[:plot_range]
                        center_x = resort(df_blade[df_blade['Point'] == point_coordinates]['Center X']).iloc[0]
                        center_y = 1079-resort(df_blade[df_blade['Point'] == point_coordinates]['Center Y']).iloc[0]
                        radius = resort(df_blade[df_blade['Point'] == point_coordinates]['Radius'])[:plot_range]
                        fps = df['fps'].iloc[0]
                        state=df['Status'].iloc[0]
                        video_number = int(df['Video Number'].iloc[0])
                        max_frame = df['Frame'].max()
                        frames_per_ms =fps/ 1000  # 每毫秒的帧数
                        frame_numbers =list(range(max_frame ))
                        x= frame_numbers / frames_per_ms
                        x=x[:plot_range]
                        if point_index==3:
                            r_el=189.3356842
                        else:
                            r_el=285.2330465
                        y1,y2,xlabel,ylabel_1,ylabel_2,unity=clac_func(plot_type,x_values1,y_values1,x_values2,y_values2,center_x,center_y,radius,fps)
                        avg1=round(np.mean(y1),2)
                        avg2=round(np.mean(y2),2)
                        amp1=round((max(y1)-min(y1))/2,3)
                        amp2=round((max(y2)-min(y2))/2,3)
                        std1=round(np.std(y1),2)
                        std2=round(np.std(y2),2)
                        contan1=lengend_func(lengend,state,blade_index,point_index,video_number,avg1,amp1,std1,unity)
                        contan2=lengend_func(lengend,state,blade_index,point_index,video_number,avg2,amp2,std2,unity)
                        x,y1,y2,xlabel,ylabel_1,ylabel_2,unity,peaks=data_process(x,y1,y2,data_pross,xlabel,ylabel_1,ylabel_2,unity,fps)
                        if plot_type =='1':
                            fig.add_subplot(nrows, 1, plot_index+1, label=f"subplot_{plot_index+1}")
                            plt.scatter(y1,y2,label=contan1)
                            plt.xlabel(f'{ylabel_1} {unity}' )
                            plt.ylabel(f'{ylabel_2} {unity}')
                            plt.legend(prop=font_prop, loc='upper right')
                            plt.grid(True)
                            plt.ylim(0,1080)
                            plt.xlim(0,1920)
                        elif plot_type =='2':
                            fig.add_subplot(nrows, 2, plot_index * 2 + 1, label=f"subplot_{plot_index*2+1}")
                            plt.plot(x,y1,label=contan1)
                            plt.xlabel(xlabel)
                            plt.ylabel(f'{ylabel_1} {unity}')
                            plt.legend(prop=font_prop, loc='upper right')
                            plt.grid(True)
                            if data_pross=='3':
                                for peak in peaks:
                                    plt.plot(x[peaks], y1[peaks], "x", label='_nolegend_', color='red')
                                    plt.annotate(f'{x[peak]:.3f} Hz', xy=(x[peak], y1[peak]), xytext=(x[peak],  y1[peak]+0.1))
                            fig.add_subplot(nrows, 2, plot_index * 2 + 2, label=f"subplot_{plot_index*2+2}")
                            plt.plot(x,y2,label=contan2)
                            plt.xlabel(xlabel)
                            plt.ylabel(f'{ylabel_2} {unity}')
                            plt.legend(prop=font_prop, loc='upper right')
                            plt.grid(True)
                            if data_pross=='3':
                                for peak in peaks:
                                    plt.plot(x[peaks], y2[peaks], "x", label='_nolegend_', color='red')
                                    plt.annotate(f'{x[peak]:.3f} Hz', xy=(x[peak], y2[peak]), xytext=(x[peak],  y1[peak]+0.1))
                        else:
                            fig.add_subplot(nrows,1, plot_index + 1, label=f"subplot_{plot_index+2}")
                            plt.plot(x,y1,label=contan1)
                            plt.xlabel(xlabel)
                            plt.ylabel(f'{ylabel_1} {unity}')
                            plt.legend(prop=font_prop, loc='upper right')
                            plt.grid(True)
                            plt.ylim(0,7)
                            if data_pross=='3':
                                for peak in peaks:
                                    plt.plot(x[peaks], y1[peaks], "x", label='_nolegend_', color='red')
                                    plt.annotate(f'{x[peak]:.3f} Hz', xy=(x[peak], y1[peak]), xytext=(x[peak],  y1[peak]+0.1))
                        
    elif group_type=='2':#葉片分組
        nrows=len(selected_blades)
        for j, blade_index in enumerate(selected_blades):
            plot_index=plot_index+1
            for i, file_info in enumerate(selected_files):
                for w,point_index in enumerate(selected_points):
                    df = pd.read_csv(os.path.join(file_path, file_info['file']), delimiter=',', low_memory=False)
                    df_blade = df[df['ROI number'] == blade_index]
                    point_coordinates = file_info[f'points{blade_index}'][point_index - 1]
                    if point_coordinates=='':
                        pass
                    else:
                        first_point_coordinates = file_info[f'points{blade_index}'][0]
                        x_values1= resort(df_blade[df_blade['Point'] == first_point_coordinates]['X Coordinate'])[:plot_range]
                        y_values1= 1079-resort(df_blade[df_blade['Point'] == first_point_coordinates]['Y Coordinate'])[:plot_range]
                        x_values2= resort(df_blade[df_blade['Point'] == point_coordinates]['X Coordinate'])[:plot_range]
                        y_values2= 1079-resort(df_blade[df_blade['Point'] == point_coordinates]['Y Coordinate'])[:plot_range]
                        center_x = resort(df_blade[df_blade['Point'] == point_coordinates]['Center X']).iloc[0]
                        center_y = 1079-resort(df_blade[df_blade['Point'] == point_coordinates]['Center Y']).iloc[0]
                        radius = resort(df_blade[df_blade['Point'] == point_coordinates]['Radius'])[:plot_range]
                        fps = df['fps'].iloc[0]
                        state=df['Status'].iloc[0]
                        video_number = int(df['Video Number'].iloc[0])
                        if point_index==3:
                            r_el=189.3356842
                        else:
                            r_el=285.2330465
                        y1,y2,xlabel,ylabel_1,ylabel_2,unity=clac_func(plot_type,x_values1,y_values1,x_values2,y_values2,center_x,center_y,radius,fps)
                        max_frame = df['Frame'].max()
                        frames_per_ms =fps/ 1000  # 每毫秒的帧数
                        frame_numbers =list(range(max_frame ))
                        x= frame_numbers / frames_per_ms
                        x=x[:plot_range]
                        avg1=round(np.mean(y1),2)
                        avg2=round(np.mean(y2),2)
                        amp1=round(max(y1)-min(y1)/2,2)
                        amp2=round(max(y2)-min(y2)/2,2)
                        std1=round(np.std(y1),2)
                        std2=round(np.std(y2),2)
                        contan1=lengend_func(lengend,state,blade_index,point_index,video_number,avg1,amp1,std1,unity)
                        contan2=lengend_func(lengend,state,blade_index,point_index,video_number,avg2,amp2,std2,unity)
                        x,y1,y2,xlabel,ylabel_1,ylabel_2,unity,peaks=data_process(x,y1,y2,data_pross,xlabel,ylabel_1,ylabel_2,unity,fps)
                        if plot_type =='1':
                            fig.add_subplot(nrows, 1, plot_index+1, label=f"subplot_{plot_index+1}")
                            plt.scatter(y1,y2,label=contan1)
                            plt.xlabel(f'{ylabel_1} {unity}' )
                            plt.ylabel(f'{ylabel_2} {unity}')
                            plt.legend(prop=font_prop, loc='upper right')
                            plt.grid(True)
                        elif plot_type =='2':
                            fig.add_subplot(nrows, 2, plot_index * 2 + 1, label=f"subplot_{plot_index*2+1}")
                            plt.plot(x,y1,label=contan1)
                            plt.xlabel(xlabel)
                            plt.ylabel(f'{ylabel_1} {unity}')
                            plt.legend(prop=font_prop, loc='upper right')
                            plt.grid(True)
                            if data_pross=='3':
                                for peak in peaks:
                                    plt.plot(x[peaks], y1[peaks], "x", label='_nolegend_', color='red')
                                    plt.annotate(f'{x[peak]:.3f} Hz', xy=(x[peak], y1[peak]), xytext=(x[peak],  y1[peak]+0.1))
                            fig.add_subplot(nrows, 2, plot_index * 2 + 2, label=f"subplot_{plot_index*2+2}")
                            plt.plot(x,y2,label=contan2)
                            plt.xlabel(xlabel)
                            plt.ylabel(f'{ylabel_2} {unity}')
                            plt.legend(prop=font_prop, loc='upper right')
                            plt.grid(True)
                            if data_pross=='3':
                                for peak in peaks:
                                    plt.plot(x[peaks], y2[peaks], "x", label='_nolegend_', color='red')
                                    plt.annotate(f'{x[peak]:.3f} Hz', xy=(x[peak], y2[peak]), xytext=(x[peak],  y1[peak]+0.1))
                        else:
                            fig.add_subplot(nrows,1, plot_index + 1, label=f"subplot_{plot_index+2}")
                            plt.plot(x,y1,label=contan1)
                            plt.xlabel(xlabel)
                            plt.ylabel(f'{ylabel_1} {unity}')
                            plt.legend(prop=font_prop, loc='upper right')
                            plt.grid(True)
                            if data_pross=='3':
                                for peak in peaks:
                                    plt.plot(x[peaks], y1[peaks], "x", label='_nolegend_', color='red')
                                    plt.annotate(f'{x[peak]:.3f} Hz', xy=(x[peak], y1[peak]), xytext=(x[peak],  y1[peak]+0.1))
    elif group_type=='3':#點分組
        nrows=len(selected_points)
        for w,point_index in enumerate(selected_points):
            plot_index=plot_index+1
            for i, file_info in enumerate(selected_files):
                for j, blade_index in enumerate(selected_blades):
                    df = pd.read_csv(os.path.join(file_path, file_info['file']), delimiter=',', low_memory=False)
                    df_blade = df[df['ROI number'] == blade_index]
                    point_coordinates = file_info[f'points{blade_index}'][point_index - 1]
                    if point_coordinates=='':
                        pass
                    else:
                        first_point_coordinates = file_info[f'points{blade_index}'][0]
                        x_values1= resort(df_blade[df_blade['Point'] == first_point_coordinates]['X Coordinate'])[:plot_range]
                        y_values1= 1079-resort(df_blade[df_blade['Point'] == first_point_coordinates]['Y Coordinate'])[:plot_range]
                        x_values2= resort(df_blade[df_blade['Point'] == point_coordinates]['X Coordinate'])[:plot_range]
                        y_values2= 1079-resort(df_blade[df_blade['Point'] == point_coordinates]['Y Coordinate'])[:plot_range]
                        center_x = resort(df_blade[df_blade['Point'] == point_coordinates]['Center X']).iloc[0]
                        center_y = 1079-resort(df_blade[df_blade['Point'] == point_coordinates]['Center Y']).iloc[0]
                        radius = resort(df_blade[df_blade['Point'] == point_coordinates]['Radius'])[:plot_range]
                        fps = df['fps'].iloc[0]
                        state=df['Status'].iloc[0]
                        video_number = int(df['Video Number'].iloc[0])
                        if point_index==3:
                            r_el=189.3356842
                        else:
                            r_el=285.2330465
                        y1,y2,xlabel,ylabel_1,ylabel_2,unity=clac_func(plot_type,x_values1,y_values1,x_values2,y_values2,center_x,center_y,radius,fps)
                        max_frame = df['Frame'].max()
                        frames_per_ms =fps/ 1000  # 每毫秒的帧数
                        frame_numbers =list(range(max_frame ))
                        x= frame_numbers / frames_per_ms
                        x=x[:plot_range]
                        avg1=round(np.mean(y1),2)
                        avg2=round(np.mean(y2),2)
                        amp1=round((max(y1)-min(y1))/2,2)
                        amp2=round((max(y2)-min(y2))/2,2)
                        std1=round(np.std(y1),2)
                        std2=round(np.std(y2),2)
                        contan1=lengend_func(lengend,state,blade_index,point_index,video_number,avg1,amp1,std1,unity)
                        contan2=lengend_func(lengend,state,blade_index,point_index,video_number,avg2,amp2,std2,unity)     
                        x,y1,y2,xlabel,ylabel_1,ylabel_2,unity,peaks=data_process(x,y1,y2,data_pross,xlabel,ylabel_1,ylabel_2,unity,fps)
                        if plot_type =='1':
                            fig.add_subplot(nrows, 1, plot_index+1, label=f"subplot_{plot_index+1}")
                            plt.scatter(y1,y2,label=contan1)
                            plt.xlabel(f'{ylabel_1} {unity}' )
                            plt.ylabel(f'{ylabel_2} {unity}')
                            plt.legend(prop=font_prop, loc='upper right')
                            plt.grid(True)
                            plt.xlim(0,1920)
                            plt.ylim(0,1080)
                        elif plot_type =='2':
                            fig.add_subplot(nrows, 2, plot_index * 2 + 1, label=f"subplot_{plot_index*2+1}")
                            plt.plot(x,y1,label=contan1)
                            plt.xlabel(xlabel)
                            plt.ylabel(f'{ylabel_1} {unity}')
                            plt.legend(prop=font_prop, loc='upper right')
                            plt.grid(True)
                            plt.ylim(y_min,y_max)
                            if data_pross=='3':
                                for peak in peaks:
                                    plt.plot(x[peaks], y1[peaks], "x", label='_nolegend_', color='red')
                                    plt.annotate(f'{x[peak]:.3f} Hz', xy=(x[peak], y1[peak]), xytext=(x[peak],  y1[peak]+0.1))
                            fig.add_subplot(nrows, 2, plot_index * 2 + 2, label=f"subplot_{plot_index*2+2}")
                            plt.plot(x,y2,label=contan2)
                            plt.xlabel(xlabel)
                            plt.ylabel(f'{ylabel_2} {unity}')
                            plt.legend(prop=font_prop, loc='upper right')
                            plt.grid(True)
                            plt.ylim(y_min,y_max)
                            if data_pross=='3':
                                for peak in peaks:
                                    plt.plot(x[peaks], y2[peaks], "x", label='_nolegend_', color='red')
                                    plt.annotate(f'{x[peak]:.3f} Hz', xy=(x[peak], y2[peak]), xytext=(x[peak],  y1[peak]+0.1))
                        else:
                            fig.add_subplot(nrows,1,plot_index  + 1, label=f"subplot_{plot_index+2}")
                            plt.plot(x,y1,label=contan1)
                            plt.xlabel(xlabel)
                            plt.ylabel(f'{ylabel_1} {unity}')
                            font_prop1=font_prop
                            font_prop1.set_size(14)
                            plt.legend(prop=font_prop1, loc='upper right')
                            plt.grid(True)
                            row_data=[f'{contan1}',f'{avg1}',f'{std1}']
                            #print(row_data)
                            if (plot_index+1) % 2 == 0:
                                plt.ylim(-50,50)
                            else:
                                plt.ylim(-25,25)
                            if data_pross=='3':
                                for peak in peaks:
                                    plt.plot(x[peaks], y1[peaks], "x", label='_nolegend_', color='red')
                                    plt.annotate(f'{x[peak]:.3f} Hz', xy=(x[peak], y1[peak]), xytext=(x[peak],  y1[peak]+0.1))                   
def main():
    global files_and_points, file_path, colors,y_min,y_max
    
    colors = plt.cm.viridis(np.linspace(0, 1, 50))
    file_path = 'D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/output_csv/5_26/'
    
    files_and_points = [
        {'file': '1_40_1.csv','points1': [3,7,8,10],'points2': [2,3,6,5],'points3': [8,6,5,4]},
        {'file': '1_40_2.csv','points1': [7,5,6,3],'points2': [7,6,5,3],'points3': [4,6,7,9]},
        {'file': '1_40_3.csv','points1': [10,6,5,4],'points2': [3,4,5,7],'points3': [7,5,4,3]},
        {'file': '1_40_4.csv','points1': [7,6,5,3],'points2': [3,5,6,8],'points3': [7,5,6,2]},
        {'file': '1_40_5.csv','points1': [8,6,3,5],'points2': [4,5,6,8],'points3': [10,6,5,3]},
        {'file': '1_40_6.csv','points1': [2,3,6,4],'points2': [8,6,5,4],'points3': [4,5,6,7]},

        {'file': '2_1_1.csv','points1': [11,6,5,4],'points2': [3,4,1,7],'points3': [2,4,5,6]},
        {'file': '2_1_2.csv','points1': [1,3,2,4],'points2': [2,4,5,7],'points3': [11,6,5,4]},
        {'file': '2_1_3.csv','points1': [8,5,4,3],'points2': ['',4,3,6],'points3': [2,4,5,7]},
        {'file': '2_1_4.csv','points1': [2,3,5,4],'points2': [6,4,3,2],'points3': [1,2,3,4]},
        {'file': '2_1_5.csv','points1': [2,'',4,5],'points2': [5,4,3,2],'points3': [1,2,3,4]},


        {'file': '3_1_1.csv','points1': [10,5,4,3],'points2': ['',2,4,6],'points3': [1,2,'',5]},
        {'file': '3_1_2.csv','points1': [2,3,4,5],'points2': [1,6,4],'points3': [5,4,3,2]},
        {'file': '3_1_3.csv','points1': [2,3,4,5],'points2': [2,'',3,4],'points3': [6,4,3,2]},
        {'file': '3_1_4.csv','points1': [11,6,5,4],'points2': [8,'',3,4],'points3': ['',2,3,4]},
        {'file': '3_1_5.csv','points1': [11,6,5,3],'points2': ['','',3,5],'points3': ['',1,2,4]},

        {'file': '4_40_6.csv','points1': [9,6,5,4],'points2': [3,'','',7],'points3': [2,3,6,4]},
        {'file': '4_40_2.csv','points1': [2,3,4,6],'points2': [9,6,'',4],'points3': [4,7,6,8]},
        {'file': '4_40_3.csv','points1': [2,3,4,5],'points2': [8,6,7,4],'points3': [7,6,5,4]},
        {'file': '4_40_4.csv','points1': [8,6,5,3],'points2': [7,6,5,3],'points3': [4,6,'',8]},
        {'file': '4_40_5.csv','points1': [2,3,4,5],'points2': [9,6,5,4],'points3': [4,5,'',7]},
        {'file': '4_40_1.csv','points1': [10,5,4,3],'points2': [2,3,6,5],'points3': [6,4,8,'']},
        {'file': '4_40_7.csv','points1': [4,5,6,7],'points2': [5,4,'',1],'points3': [8,6,5,3]},
    ]

    print("可選擇的文件名：")
    for i, item in enumerate(files_and_points, 1):
        print(f"{i}. {item['file']}")
    file_selection = input("請選擇文件（输入对应文件前面的数字，多个文件用逗号分隔）：")
    file_indices = [int(idx.strip()) - 1 for idx in file_selection.split(',')]
    # 获取用户选择的文件名列表
    selected_files = [files_and_points[idx]['file'] for idx in file_indices]
    print(f"你選擇了文件：{', '.join(selected_files)}")
    selected_files = [files_and_points[idx] for idx in file_indices]
    blades = input("請選擇要分析的 blade（輸入数字，多個 blade 用逗號分隔）：")
    selected_blades = [int(blade.strip()) for blade in blades.split(',')]

    points = input("請選擇要分析的點（輸入数字，多個點用逗號分隔）：")
    selected_points = [int(point.strip()) for point in points.split(',')]

    plot_type = input("請選擇要分析的內容：\n1.x,y散點圖 2.x,y對時間圖 3.轉速對時間 4.edgewise 5.flapwise 6.elongation\n選擇：")
    data_pross = input("請選擇要使用的數據處理方式：\n1. normalize 2.standardize 3.FFT\n選擇：")
    group_type= input('可選擇分組模式：1.文件分組 2.葉片分組 3.點分組\n選擇：')
    lengend=input("請選擇圖例內容（輸入数字，多個內容用逗號分隔）\n1.狀態 2.葉片編號 3.點編號 4.數據集編號")
    lengend = [int(x) for x in lengend.split(',')]
    y_min,y_max=0,7
    group_func(selected_files,selected_blades,selected_points,plot_type,data_pross,group_type,lengend)
    
    plt.suptitle(f'三片受損-{title}{pr_type}',fontproperties=font_prop,fontsize=20)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2, wspace=0.5)
    plt.subplots_adjust(left=0.05, right=0.96, top=0.9, bottom=0.08)
    plt.show()
if __name__ == "__main__":
    main()
