import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import pandas as pd
import csv

def on_mouse(event,x,y,flags,param):
    global old_gray,point1,point2,cut_img,min_x,min_y
    img2=old_gray.copy()
    if event==cv.EVENT_LBUTTONDOWN:
        point1=(x,y)
        cv.circle(img2,point1,10,(0,255,0),1)
        cv.imshow('image',img2)

    elif event==cv.EVENT_MOUSEMOVE and (flags&cv.EVENT_FLAG_LBUTTON):
        cv.rectangle(img2,point1,(x,y),(255,0,0),1)
        cv.imshow('image',img2)

    elif event==cv.EVENT_LBUTTONUP:#左键释放
        point2=(x,y)
        cv.rectangle(img2,point1,point2,(0,0,255),1)
        cv.imshow('image',img2)
        min_x=min(point1[0],point2[0])
        min_y=min(point1[1],point2[1])
        width=abs(point1[0]-point2[0])
        height=abs(point1[1]-point2[1])
        cut_img=old_gray[min_y:min_y+height,min_x:min_x+width]

def main():
    #設定參數
    global old_gray,cut_img
    sift1_params = dict(nfeatures = 5000,
                   nOctaveLayers = 3,
                   contrastThreshold = 0.04,
                   edgeThreshold = 10,
                   sigma =1.6 )
    sift2_params = dict(nfeatures = 5000,
                   nOctaveLayers = 3,
                   contrastThreshold = 0.04,
                   edgeThreshold = 10,
                   sigma =1.6 )
    surf1_params = dict(hessianThreshold= 13000 ,
                   nOctaves= 3,
                   nOctaveLayers =3)
    surf2_params = dict(hessianThreshold= 13000 ,
                   nOctaves= 3,
                   nOctaveLayers =3)
# Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (20,20),       
                  maxLevel = 5,            
                  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.01))                                              
    cap = cv.VideoCapture("D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/test_video/5_04/504_3_17.MOV")

    output_path='D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/output_video/17--.mp4' #影片輸出地址
    csv_filename = 'D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/output_csv/5_04/3_17_2.csv' #csv輸出地址
    ret, old_frame = cap.read()
    old_gray = old_frame

    #影片輸出參數
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))    
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) 
    fourcc = cv.VideoWriter_fourcc(*'mp4v')          
    fps=120                  
     
    out = cv.VideoWriter(output_path, fourcc,fps,(width,  height)) 

    cv.namedWindow('image')
    cv.setMouseCallback('image',on_mouse)
    cv.imshow('image',old_gray)
    cv.waitKey(0)
    
    start=time.process_time()
    sift1 = cv.xfeatures2d.SIFT_create(**sift1_params)
    sift2 = cv.xfeatures2d.SIFT_create(**sift2_params)
    surf1 = cv.xfeatures2d.SURF_create(**surf1_params)
    surf2 = cv.xfeatures2d.SURF_create(**surf2_params)

    kp2, des2 = surf2.detectAndCompute(cut_img, None)

    matched_points_img2=cv.KeyPoint_convert(kp2) #特徵描述子轉換成座標

    matched_points_img2[:, 0] += min_x
    matched_points_img2[:, 1] += min_y
    p0=matched_points_img2.reshape(-1, 1, 2) 
    # 将坐标数据转换为 NumPy 数组
    coordinates_array = np.array(p0)

    # 使用 numpy.unique 函数将重复的坐标筛除
    p0 = np.unique(coordinates_array, axis=0)

# Create some random colors(繪製光流軌跡線與點用)
    color = np.random.randint(0,255,(np.shape(p0)[0],3))
    old_gray = cv.cvtColor(old_gray, cv.COLOR_BGR2GRAY)
# Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    mask[mask==0] ='255'
    frame_number=1
    tp2=[]
    while(1):
        ret,frame = cap.read()

    #如果讀不到圖片則退出
        if frame is None:
            break
        frame_number=frame_number+1
    #將影像轉為灰度圖片
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.GaussianBlur(frame_gray,(3,3),0)
    # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params) 
    # draw the tracks(i個角點)
        for i,(new,old) in enumerate(zip(p1, p0)):
            a,b = new.ravel() # 新點
            c,d = old.ravel() # 舊點
            a, b, c, d = int(a), int(b), int(c), int(d)

            if st[i]==1:
                color[i]=(0,255,0)
            else:
                color[i]=(0,0,255)
            
            frame = cv.circle(frame,(a,b),2,color[i].tolist(),-1)
            winSize=lk_params['winSize']
            win_x=int(winSize[0]/2)
            win_y=int(winSize[1]/2)
            tracks = cv.line(mask, (a,b),(c,d), color[i].tolist(), 1)
            frame = cv.putText(frame,'Point'+str(i),
                       (a,b), cv.FONT_HERSHEY_SIMPLEX,0.35, color[i].tolist(), 1, cv.LINE_AA)
            frame = cv.rectangle(frame, (c-win_x, d-win_y), (c+win_x,d+win_y), color[i].tolist(), 1)  
        tracks_gray = cv.cvtColor(tracks, cv.COLOR_BGR2GRAY)  
        ret, mask1  = cv.threshold(tracks_gray, 200, 255, cv.THRESH_BINARY_INV)
        path = cv.bitwise_and(tracks,tracks, mask = mask1 )
        ret, mask2  = cv.threshold(tracks_gray, 200, 255, cv.THRESH_BINARY)
        bg = cv.bitwise_and(frame, frame, mask = mask2 )
        img = cv.add(bg,path)
    #展示並輸出
        cv.imshow('frame',img)
        out.write(img)
    #按esc鍵停止迴圈
        k = cv.waitKey(2) & 0xff
        if k == 27:
            break
    # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        tp1=p1.reshape(-1,2)
        tp2.append(tp1)
        p0 = p1.reshape(-1,1,2)
    tp2=np.array(tp2)

    # 將 tp2 寫入 CSV 檔案
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # 寫入 CSV 文件的標題行
        writer.writerow(['Frame', 'Point', 'X Coordinate', 'Y Coordinate'])
        
        # 遍歷 tp2 中的每一個幀
        for frame_idx, frame in enumerate(tp2, start=1):
            # 遍歷幀中的每一個點
            for point_idx, point in enumerate(frame, start=1):
                # 將幀數、點索引、x座標和y座標寫入 CSV 文件中
                writer.writerow([frame_idx, point_idx, point[0], point[1]])

    print("CSV file saved successfully:", csv_filename)

    # 提取 10 个点的 x 和 y 坐标
    num_points = tp2.shape[1]
    x_points = [tp2[:, i, 0] for i in range(num_points)]
    y_points = [tp2[:, i, 1] for i in range(num_points)]

# 创建子图
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

# 在第一个子图中绘制所有点的 x 坐标
    for i, x in enumerate(x_points):
        axs[0].plot(range(len(x)), x, label=f'X Coordinate of Point {i+1}')

# 设置第一个子图的标题和标签
    axs[0].set_title('X Coordinate of Points 1-10')
    axs[0].set_xlabel('Frame Number')
    axs[0].set_ylabel('X Coordinate')
    axs[0].legend()  # 添加图例
    axs[0].grid(True)  # 显示网格线

    # 在第二个子图中绘制所有点的 y 坐标
    for i, y in enumerate(y_points):
        axs[1].plot(range(len(y)), y, label=f'Y Coordinate of Point {i+1}')

    # 设置第二个子图的标题和标签
    axs[1].set_title('Y Coordinate of Points 1-10')
    axs[1].set_xlabel('Frame Number')
    axs[1].set_ylabel('Y Coordinate')
    axs[1].legend()  # 添加图例
    axs[1].grid(True)  # 显示网格线

    plt.tight_layout()
    plt.show()  

    # 创建三维图形对象
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制每个点的三维坐标
    for i in range(num_points):
        x = tp2[:, i, 0]  # 提取第 i 个点的 x 坐标
        z = tp2[:, i, 1]  # 提取第 i 个点的 y 坐标
        y = range(len(x))  # 平面上的 z 坐标为幀数
        ax.scatter(x, y, z, label=f'Point {i+1}')

    # 设置图形标题和标签
    ax.set_title('3D Point Cloud')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Frame Number')
    ax.legend()  # 添加图例

    plt.show()
    
    cap.release()
    out.release()
    cv.destroyAllWindows()

if __name__=='__main__':
    main()
