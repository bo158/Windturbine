import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import csv
# 初始化全局變量
point1 = None
point2 = None
rectangles = []
cut_images = []
min_x_values = []
min_y_values = []


def on_mouse(event, x, y, flags, param):
    global point1, point2, rectangles
    if event == cv.EVENT_LBUTTONDOWN:
        point1 = (x, y)
        img=resized_img.copy()
        cv.circle(img, point1, 10, (0, 255, 0), 1)
        cv.imshow('image',img)
    elif event == cv.EVENT_MOUSEMOVE and (flags & cv.EVENT_FLAG_LBUTTON):
        img=resized_img.copy()
        cv.rectangle(img, point1, (x, y), (255, 0, 0), 1)
        cv.imshow('image',img)
    elif event == cv.EVENT_LBUTTONUP:  # 左键释放
        point2 = (x, y)
        img=resized_img.copy()
        cv.rectangle(img, point1, point2, (0, 0, 255), 1)
        cv.imshow('image',img)
def process_rectangles():
    global cut_images
    min_x = min(point1[0], point2[0]) * 2
    min_y = min(point1[1], point2[1]) * 2
    width = abs(point1[0] - point2[0]) * 2
    height = abs(point1[1] - point2[1]) * 2
    cut_img = first_frame[min_y:min_y + height, min_x:min_x + width]
    cut_images.append(cut_img)
    min_x_values.append(min_x)
    min_y_values.append(min_y)
def initialize_detector(detector_type, sift_params=None, surf_params=None):
    if detector_type == 'sift':
        return cv.xfeatures2d.SIFT_create(**(sift_params if sift_params else {}))
    elif detector_type == 'surf':
        return cv.xfeatures2d.SURF_create(**(surf_params if surf_params else {}))
    elif detector_type == 'orb':
        return cv.ORB_create(10)
    else:
        raise ValueError("無效的特徵檢測器選擇")
# 特徵點檢測函數
def detect_keypoints(detector, image):
    keypoints = detector.detect(image, None)
    keypoints = cv.KeyPoint_convert(keypoints)
    return keypoints

def pad_to_desired_shape(image, desired_shape, fill_value=0):
    if image.shape == desired_shape:
        return image
    pad_width = [(max(desired_shape[i] - image.shape[i], 0), 0) for i in range(len(desired_shape))]
    # 使用np.pad()函数填充数组
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=fill_value)
    return padded_image
def main():
    #設定參數
    global resized_img,first_frame
    sift_params = dict(nfeatures = 10,
                   nOctaveLayers = 3,
                   contrastThreshold = 0.04,
                   edgeThreshold = 20,
                   sigma =1.6 )
    surf_params = dict(hessianThreshold= 3500,
                   nOctaves= 3,
                   nOctaveLayers =3)
# Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (8,8),       
                  maxLevel = 3,            
                  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 50, 0.01))  
    circle_img_range=10  
    desired_shape = (circle_img_range*2,circle_img_range*2, 3)
                                          
    cap = cv.VideoCapture('D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/test_video/5_26/IMG_1893.MOV')
    csv_filename = 'D:/dfg22/Documents/MEGA/NOTE/Mechical/project_code/output_csv/5_26/1_40_3.csv' #csv輸出地址
    feature_detector_type = input('選擇特徵檢測器 (sift, surf, orb): ').strip().lower()

    ret, first_frame = cap.read()
    frame_gray= cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
    img=first_frame.copy()
    resized_img =cv.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    #影片輸出參數
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))    
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) 
    fourcc = cv.VideoWriter_fourcc(*'mp4v')          
    fps=cap.get(cv.CAP_PROP_FPS)                  

    cv.namedWindow('image')
    cv.setMouseCallback('image',on_mouse)
    cv.imshow('image',resized_img)
    while True:
        key = cv.waitKey(1)
        if key == 13:  # Enter key
            if not rectangles or rectangles[len(cut_images)-1][0]!=point1 :
                cv.rectangle(resized_img, point1, point2, (0, 0, 255), 1)
                cv.imshow('image',resized_img)
                process_rectangles()
                print(f"已框選ROI數量: {len(cut_images)}")
                rectangles.append((point1, point2))
            else:
                print('請框選新ROI')
        elif key == 27:  # Esc key
            break 
    start=time.process_time()
    detector = initialize_detector(feature_detector_type, sift_params, surf_params)
    p0 = np.zeros((0,1, 2))
    circle_img_list=[]
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Frame','ROI number', 'Point', 'X Coordinate', 'Y Coordinate','fixed X Coordinate','fixed Y Coordinate','Radius','Center X','Center Y','fixed Center X','fixed Center Y','fps','Width','Height','Status','Date','Fan Speed','Video Number'])
    for i in range(len(cut_images)):
        kp = detect_keypoints(detector, cut_images[i])
        kp[:, 0] += min_x_values[i]
        kp[:, 1] += min_y_values[i]
        kp2 = kp.reshape(-1, 1, 2) 
        coordinates_array = np.array(kp2)
        kp2 = np.unique(coordinates_array, axis=0)
        circle_img_list.append(len(kp2))
        p0 = np.concatenate((p0, kp2), axis=0)
    arrays = [np.arange(1, count + 1) for count in circle_img_list]
    point_list = np.concatenate(arrays)
    p0 = p0.astype('float32')
    circle_img_number = np.repeat(np.arange(1, len(circle_img_list) + 1), circle_img_list)
    for i,old in enumerate(p0):
        a,b = old.ravel()
        a_1,b_1=a,b
        a,b=int(a),int(b)
        circle_img_gray = frame_gray[b-circle_img_range:b+circle_img_range, a-circle_img_range:a+circle_img_range]
        circle_img=frame_gray[b-circle_img_range:b+circle_img_range, a-circle_img_range:a+circle_img_range]
        _, thresh = cv.threshold(circle_img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # 二值化处理
        thresh = cv.bitwise_not(thresh)  # 反转二值图像
        element = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(2,2))  # 获取形态学操作所需的结构元素
        morph_img = thresh.copy()  # 复制二值图像
        cv.morphologyEx(src=thresh, op=cv.MORPH_CLOSE, kernel=element, dst=morph_img)  # 闭运算，填充物体内部的空洞
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
        with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([1,circle_img_number[i],point_list[i], a_1,b_1,a-circle_img_range+x, b-circle_img_range+y, radius,'','','','','','','','','','',''])
# Create some random colors(繪製光流軌跡線與點用)
    end_surf=time.process_time()
    surf_time = end_surf-start
    print(f'特徵檢索時間：{surf_time}s')
    color=np.zeros((len(p0),3))
    old_gray = cv.cvtColor(first_frame, cv.COLOR_BGR2GRAY)
# Create a mask image for drawing purposes
    mask = np.zeros_like(first_frame)
    mask[mask==0] ='255'
    frame_number=1
    circle_img_list = [None] * len(p0)
    while(1):
        ret,frame = cap.read()
    #如果讀不到圖片則退出
        if frame is None:
            break
        frame_number=frame_number+1
    #將影像轉為灰度圖片
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        #frame_gray = cv.GaussianBlur(frame_gray,(3,3),0)
    # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray,p0, None, **lk_params) 
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
            circle_img_gray = frame_gray[b-circle_img_range:b+circle_img_range, a-circle_img_range:a+circle_img_range]
            circle_img=frame[b-circle_img_range:b+circle_img_range, a-circle_img_range:a+circle_img_range]

            thresh =cv.adaptiveThreshold(circle_img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 19, 2)   # 二值化处理
            thresh = cv.bitwise_not(thresh)  # 反转二值图像
            element = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(2,2))  # 获取形态学操作所需的结构元素
            if thresh is not None:
                morph_img = thresh.copy()  # 复制二值图像
                cv.morphologyEx(src=thresh, op=cv.MORPH_CLOSE, kernel=element, dst=morph_img)  # 闭运算，填充物体内部的空洞
                circle_img_gray = np.stack((circle_img_gray,) * 3, axis=-1)
                thresh = np.stack((thresh,) * 3, axis=-1)
                morph_img_3= np.stack((morph_img,) * 3, axis=-1)
                padded_image_gray = pad_to_desired_shape(circle_img_gray, desired_shape)
                padded_thresh = pad_to_desired_shape(thresh, desired_shape)
                padded_morph_img_3 = pad_to_desired_shape(morph_img_3, desired_shape)
                padded_image = pad_to_desired_shape(circle_img, desired_shape)
                combined_img = np.concatenate((padded_image_gray,padded_thresh,padded_morph_img_3,padded_image), axis=0)
                circle_img_list[i] =  combined_img
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
                    cv.circle(circle_img_list[i], center, radius_int, (0, 255, 0), 1)
                    frame=cv.circle(frame, center, radius_int, (0, 255, 0), 1)
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([frame_number,circle_img_number[i], point_list[i], a_1,b_1,a-circle_img_range+x, b-circle_img_range+y, radius,'','','','','','','','','','',''])
            frame = cv.circle(frame,(a,b),2,color[i].tolist(),-1)
            winSize=lk_params['winSize']
            win_x=int(winSize[0]/2)
            win_y=int(winSize[1]/2)
            tracks = cv.line(mask, (a,b),(c,d), color[i].tolist(), 1)
            frame = cv.putText(frame,'Point'+str(i),
                       (a,b), cv.FONT_HERSHEY_SIMPLEX,0.35, color[i].tolist(), 1, cv.LINE_AA)
            frame = cv.rectangle(frame, (c-win_x, d-win_y), (c+win_x,d+win_y), color[i].tolist(), 1) 
        combined_circle_img = np.concatenate(circle_img_list, axis=1)
        if combined_circle_img.shape[1] > 0:  # 如果 ROI 的宽度和高度都大于 0
                cv.imshow('combined_circle_img', combined_circle_img)
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
        bg1 = cv.bitwise_and(frame, frame, mask = mask2 )
        frame_gray1=np.stack((frame_gray,) * 3, axis=-1)
        bg2 = cv.bitwise_and(frame_gray1, frame_gray1, mask = mask2 )
        img1 = cv.add(bg1,path)
        img2 = cv.add(bg2,path)
        combined_img1 = np.concatenate((img1,img2), axis=0)
        resized_img1 =cv.resize(combined_img1, (combined_img1.shape[1] // 2, combined_img1.shape[0] // 2))

        cv.imshow('frame',resized_img1)
    #按esc鍵停止迴圈
        k = cv.waitKey(2) & 0xff
        if k == 27:
            break
    # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = p1.reshape(-1,1,2)
    end_optical_flow=time.process_time()
    print(f'光流花費時間：{end_optical_flow-end_surf}s')
    cv.destroyAllWindows()
    status = input('風扇狀態?')
    date = input('日期：(ex:05/20)')
    fan_speed = input('風扇速度?')
    video_number = input('第幾部影片?')
    print("CSV file saved successfully:", csv_filename)   
    with open(csv_filename, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)
        rows[1][12:19] = [fps,width,height,status,date,fan_speed,video_number]
    with open(csv_filename, mode='w', newline='',encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(rows)
    cap.release()
if __name__=='__main__':
    main()
