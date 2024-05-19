import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import csv
def on_mouse(event,x,y,flags,param):
    global point1,point2,cut_img,min_x,min_y
    img2=resized_img.copy()
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
        min_x=min(point1[0],point2[0])*2
        min_y=min(point1[1],point2[1])*2
        width=abs(point1[0]-point2[0])*2
        height=abs(point1[1]-point2[1])*2
        cut_img=old_frame[min_y:min_y+height,min_x:min_x+width]
def main():
    #設定參數
    global cut_img,resized_img,old_frame
    sift_params = dict(nfeatures = 30,
                   nOctaveLayers = 3,
                   contrastThreshold = 0.04,
                   edgeThreshold = 20,
                   sigma =1.6 )
    surf_params = dict(hessianThreshold= 13000 ,
                   nOctaves= 3,
                   nOctaveLayers =3)
# Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),       
                  maxLevel = 3,            
                  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 50, 0.01))                                              
    cap = cv.VideoCapture('D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/test_video/5_10/510_1_27_1.MOV')
    csv_filename = 'D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/output_csv/5_10/4_24_1_1.csv' #csv輸出地址
    
    ret, old_frame = cap.read()
    img=old_frame.copy()
    resized_img =cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    #影片輸出參數
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))    
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) 
    fourcc = cv.VideoWriter_fourcc(*'mp4v')          
    fps=cap.get(cv.CAP_PROP_FPS)                  

    cv.namedWindow('image')
    cv.setMouseCallback('image',on_mouse)
    cv.imshow('image',resized_img)
    cv.waitKey(0)
    
    start=time.process_time()
    sift= cv.xfeatures2d.SIFT_create(**sift_params)
    surf= cv.xfeatures2d.SURF_create(**surf_params)
    orb = cv.ORB_create(30)
    kp1 = surf.detect(cut_img, None)

    matched_points_img2=cv.KeyPoint_convert(kp1) #特徵描述子轉換成座標

    matched_points_img2[:, 0] += min_x
    matched_points_img2[:, 1] += min_y
    p0=matched_points_img2.reshape(-1, 1, 2) 
    # 将坐标数据转换为 NumPy 数组
    coordinates_array = np.array(p0)

    # 使用 numpy.unique 函数将重复的坐标筛除
    p0 = np.unique(coordinates_array, axis=0)

# Create some random colors(繪製光流軌跡線與點用)
    color = np.random.randint(0,255,(np.shape(p0)[0],3))
    old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
# Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)
    mask[mask==0] ='255'
    frame_number=1
    tp2=[]
    roi_list = [None] * len(p0)
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame', 'Point', 'X Coordinate', 'Y Coordinate','Radius','X center','Y center','fps'])
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
            a_1,b_1=a,b
            c,d = old.ravel() # 舊點
            a, b, c, d = int(a), int(b), int(c), int(d)
            if st[i]==1:
                color[i]=(0,255,0)
            else:
                color[i]=(0,0,255)
            roi_range=20
            roi = frame_gray[b-roi_range:b+roi_range, a-roi_range:a+roi_range]
            roi_1=frame[b-roi_range:b+roi_range, a-roi_range:a+roi_range]
            _, thresh = cv.threshold(roi, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # 二值化处理
            thresh = cv.bitwise_not(thresh)  # 反转二值图像
            element = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(2,2))  # 获取形态学操作所需的结构元素
            morph_img = thresh.copy()  # 复制二值图像
            cv.morphologyEx(src=thresh, op=cv.MORPH_CLOSE, kernel=element, dst=morph_img)  # 闭运算，填充物体内部的空洞
            roi_list[i] =  roi_1
            _,contours, _ = cv.findContours(morph_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # 检测图像中的轮廓       
# 如果有检测到轮廓
            if contours:
                # 初始化圆形对象列表
                circles = []
                # 对每个轮廓进行处理
                areas = [cv.contourArea(c) for c in contours]  # 计算轮廓的面积
                sorted_areas = np.sort(areas)  # 对面积进行排序
                cnt = contours[areas.index(sorted_areas[-1])]  # 获取最大面积的轮廓
                # 绘制最小外接圆（绿色）
                (x, y), radius = cv.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius_int = int(radius)
                cv.circle(roi_list[i], center, radius_int, (0, 255, 0), 1)
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_number-1, i+1, a_1,b_1, radius,a-roi_range+x, b-roi_range+y,fps])
            frame = cv.circle(frame,(a,b),2,color[i].tolist(),-1)
            winSize=lk_params['winSize']
            win_x=int(winSize[0]/2)
            win_y=int(winSize[1]/2)
            tracks = cv.line(mask, (a,b),(c,d), color[i].tolist(), 1)
            frame = cv.putText(frame,'Point'+str(i),
                       (a,b), cv.FONT_HERSHEY_SIMPLEX,0.35, color[i].tolist(), 1, cv.LINE_AA)
            frame = cv.rectangle(frame, (c-win_x, d-win_y), (c+win_x,d+win_y), color[i].tolist(), 1) 
        combined_roi = np.concatenate(roi_list, axis=0)
        if roi.shape[0] > 0 and combined_roi.shape[1] > 0:  # 如果 ROI 的宽度和高度都大于 0
                cv.imshow('combined_roi', combined_roi)
        else:
                print("ROI is empty or has size 0")
        frame = cv.putText(frame,'SIFT point: '+str(i+1)+'   '+'SIFT Parameters '+str(list(surf_params.items())),
                       (10,20), cv.FONT_HERSHEY_SIMPLEX,0.35, (0,0,0), 1, cv.LINE_AA)
        frame = cv.putText(frame,'LK Parameters '+str(list(lk_params.items())),
                       (10,35), cv.FONT_HERSHEY_SIMPLEX,0.35, (0,0,0), 1, cv.LINE_AA)
        frame = cv.putText(frame,'Frame:'+str(frame_number)+' fps:'+str(fps),
                       (10,50), cv.FONT_HERSHEY_SIMPLEX,0.35, (0,0,0), 1, cv.LINE_AA) 
        tracks_gray = cv.cvtColor(tracks, cv.COLOR_BGR2GRAY)  
        ret, mask1  = cv.threshold(tracks_gray, 200, 255, cv.THRESH_BINARY_INV)
        path = cv.bitwise_and(tracks,tracks, mask = mask1 )
        ret, mask2  = cv.threshold(tracks_gray, 200, 255, cv.THRESH_BINARY)
        bg = cv.bitwise_and(frame, frame, mask = mask2 )
        img = cv.add(bg,path)
    #展示
        img=cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
        cv.imshow('frame',img)
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
    print("CSV file saved successfully:", csv_filename)   
    cap.release()
    cv.destroyAllWindows()

if __name__=='__main__':
    main()
