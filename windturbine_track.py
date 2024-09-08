import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import time
import csv
import tkinter as tk
from tkinter import filedialog,messagebox
from tkinter.filedialog import askopenfilename
from tqdm import tqdm
from tkcalendar import DateEntry

# 初始化全局變量
point1 = None
point2 = None
rectangles = []
cut_images = []
min_x_values = []
min_y_values = []
def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw() 
    try:
        filepath = askopenfilename(title=title, filetypes=filetypes)
    finally:
        # 确保 Tkinter 资源被销毁
        root.destroy()
    return filepath

def select_save_path(title,file_types):
    # 创建 Tkinter 主窗口并隐藏
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    # 打开保存文件对话框
    try:
        filepath = filedialog.asksaveasfilename(title=title,filetypes=file_types)
    finally:
        root.destroy()
    return filepath

def create_gui():
    def submit_form():
        global result_data
        # 获取用户输入
        fan_status = fan_status_entry.get()
        date = date_entry.get_date()  # 从 DateEntry 获取日期
        fan_speed = fan_speed_entry.get()
        video_number = video_number_entry.get()

        # 存储结果数据
        result_data = (csv_filename, fan_status, date.strftime('%m/%d/%Y'), fan_speed, video_number)
        root.destroy()  # 关闭窗口并退出程序

    def prompt_for_data():
        # 创建主窗口
        global root, fan_status_entry, date_entry, fan_speed_entry, video_number_entry
        root = tk.Tk()
        root.title("數據輸入")

        # 创建并放置标签和输入框
        tk.Label(root, text="葉片狀態:").grid(row=0, column=0, padx=10, pady=10)
        fan_status_entry = tk.Entry(root)
        fan_status_entry.grid(row=0, column=1, padx=10, pady=10)

        tk.Label(root, text="日期:").grid(row=1, column=0, padx=10, pady=10)
        date_entry = DateEntry(root, date_pattern='mm/dd/yyyy')
        date_entry.grid(row=1, column=1, padx=10, pady=10)

        tk.Label(root, text="風機速度:").grid(row=2, column=0, padx=10, pady=10)
        fan_speed_entry = tk.Spinbox(root, from_=0, to=100, increment=1)
        fan_speed_entry.grid(row=2, column=1, padx=10, pady=10)

        tk.Label(root, text="影片編號:").grid(row=3, column=0, padx=10, pady=10)
        video_number_entry = tk.Spinbox(root, from_=1, to=100, increment=1)
        video_number_entry.grid(row=3, column=1, padx=10, pady=10)

        # 创建提交按钮
        submit_button = tk.Button(root, text="提交", command=submit_form)
        submit_button.grid(row=4, columnspan=2, padx=10, pady=10)

        # 运行主循环
        root.mainloop()

    # 首先选择保存路径
    global csv_filename, result_data
    csv_filename = select_save_path('保存數據文件',[("CSV files", "*.csv")])

    # 如果用户没有选择文件路径，则退出程序
    if not csv_filename:
        result_data = None
        return result_data

    # 显示输入表单
    prompt_for_data()

    return result_data
def create_selection_gui():
    def submit_selection():
        global feature_detector_type, find_circle
        feature_detector_type = feature_detector_var.get()
        find_circle = find_circle_var.get()

        print(f"選擇特徵檢測器: {feature_detector_type}")
        print(f"是否尋找圓形目標: {find_circle}")

        # 关闭窗口
        root.destroy()

    # 创建主窗口
    global root, feature_detector_var, find_circle_var
    root = tk.Tk()
    root.title("選擇特徵檢測器")

    # 创建特征检测器选择
    tk.Label(root, text="選擇特徵檢測器:").pack(padx=10, pady=10)
    feature_detector_var = tk.StringVar(value="sift")
    detectors = ["sift", "surf", "orb", "shi"]
    for detector in detectors:
        tk.Radiobutton(root, text=detector.upper(), variable=feature_detector_var, value=detector).pack(anchor=tk.W)

    # 创建寻找圆形目标选项
    find_circle_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="尋找圓形目標", variable=find_circle_var).pack(padx=10, pady=10)

    # 创建提交按钮
    submit_button = tk.Button(root, text="提交", command=submit_selection)
    submit_button.pack(pady=20)

    # 运行主循环
    root.mainloop()

    return feature_detector_type, find_circle
def resize_image_for_screen(img, screen_width, screen_height):
    # 获取图像的原始尺寸
    original_height, original_width = img.shape[:2]

    # 计算缩放因子，保持长宽比
    width_ratio = screen_width / original_width
    height_ratio = screen_height / original_height
    scaling_factor = min(width_ratio, height_ratio)

    # 计算新的尺寸
    new_width = int(original_width * scaling_factor)
    new_height = int(original_height * scaling_factor)

    # 调整图像大小
    resized_img = cv.resize(img, (new_width, new_height))

    # 返回调整后的图像和缩放因子
    return resized_img, scaling_factor

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
def process_rectangles(scaling_factor):
    global cut_images
    min_x = min(point1[0], point2[0]) * (1/scaling_factor)
    min_y = min(point1[1], point2[1]) * (1/scaling_factor)
    width = abs(point1[0] - point2[0]) * (1/scaling_factor)
    height = abs(point1[1] - point2[1]) * (1/scaling_factor)
    min_x = int(min_x)
    min_y = int(min_y)
    width = int(width)
    height = int(height)
    cut_img = first_frame[min_y:min_y + height, min_x:min_x + width]
    cut_images.append(cut_img)
    min_x_values.append(min_x)
    min_y_values.append(min_y)
def initialize_detector(detector_type, sift_params=None, surf_params=None,orb_params=None):
    if detector_type == 'sift':
        if version.startswith('3.'):
            return cv.xfeatures2d.SIFT_create(**(sift_params if sift_params else {}))
        elif version.startswith('4.'):
            return cv.SIFT_create(**(sift_params if sift_params else {}))
        
    elif detector_type == 'surf':
        if version.startswith('3.'):
            return cv.xfeatures2d.SURF_create(**(surf_params if surf_params else {}))
        elif version.startswith('4.'):
            print("現存版本不可用，改用默認選項SIFT" )
            return cv.SIFT_create(**(sift_params if sift_params else {}))
    elif detector_type == 'orb':
        return cv.ORB_create(**(orb_params if surf_params else {}))
    elif detector_type == 'shi':
        return 0
    else:
        raise ValueError("無效的特徵檢測器選擇")
# 特徵點檢測函數
def detect_keypoints(detector, image,detector_type):
    if detector_type == 'shi':
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        keypoints = cv.goodFeaturesToTrack(image_gray,50,0.01,10)
        keypoints = keypoints.reshape(-1, 2)
    else:
        keypoints = detector.detect(image, None)
        keypoints = cv.KeyPoint_convert(keypoints)
    return keypoints

def pad_to_desired_shape(image, desired_shape, fill_value=0):
    if image.shape == desired_shape:
        return image
    if len(image.shape) == 2:
        # 将灰度图像转换为 RGB 彩色图像
        image= np.stack([image] * 3, axis=-1)
    
    pad_width = [(max(desired_shape[i] - image.shape[i], 0), 0) for i in range(len(desired_shape))]
    # 使用np.pad()函数填充数组
    padded_image = np.pad(image, pad_width, mode='constant', constant_values=fill_value)
    return padded_image
def crop_center(image, crop_x, crop_y):
    y, x, _ = image.shape
    start_x = x // 2 - (crop_x // 2)
    start_y = y // 2 - (crop_y // 2)
    return image[start_y:start_y+crop_y, start_x:start_x+crop_x]
def img_preprocess(img):
    #image = cv.GaussianBlur(img, (3, 3), 0)
    image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #image = cv.bilateralFilter(image, d=5, sigmaColor=10, sigmaSpace=10)
    #image = cv.medianBlur(image,5)
    #image = cv.Canny(image, 170, 170)
    #thre, image= cv.threshold(image, 125, 255, cv.THRESH_BINARY)
    return image
def find_circle_process(find_circle,frame_gray,circle_img_range,a,b):
    x, y, radius=0,0,0
    combined_img = None
    desired_shape = (circle_img_range*2,circle_img_range*2, 3)
    if find_circle:
            circle_img_gray = frame_gray[b-circle_img_range:b+circle_img_range, a-circle_img_range:a+circle_img_range]
            circle_img=frame_gray[b-circle_img_range:b+circle_img_range, a-circle_img_range:a+circle_img_range]
            #thresh = cv.adaptiveThreshold(circle_img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 1)   # 自动阈值二值化
            #_, thresh = cv.threshold(circle_img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # 二值化处理
            _, thresh = cv.threshold(circle_img_gray,30, 255, cv.THRESH_BINARY)
            #thresh = cv.adaptiveThreshold(circle_img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
            
            thresh = cv.bitwise_not(thresh)  # 反转二值图像
            element = cv.getStructuringElement(shape=cv.MORPH_RECT, ksize=(1,1))  # 获取形态学操作所需的结构元素
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
  
                if version.startswith('3.'):
                    _,contours, _ = cv.findContours(morph_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # 检测图像中的轮廓  
                elif version.startswith('4.'):
                    contours, _ = cv.findContours(morph_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)  # 检测图像中的轮廓     
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
    return x, y, radius,combined_img

def main():
    #設定參數
    global resized_img,first_frame,version
    version = cv.__version__
    screen_width=1920 
    screen_height=700
    scale_factor=2
    sift_params = dict(nfeatures = 25,
                   nOctaveLayers = 5,
                   contrastThreshold = 0.04,
                   edgeThreshold = 20,
                   sigma =1.6 )
    surf_params = dict(hessianThreshold= 6000,
                   nOctaves= 3,
                   nOctaveLayers =3)
    orb_params = dict(nfeatures=10,         # 最大特徵點數，預設值
                scaleFactor=1.2,       # 圖像金字塔的縮放因子，預設值
                nlevels=8,             # 金字塔層數，預設值
                edgeThreshold=31,      # 邊緣閾值，預設值
                firstLevel=0,          # 金字塔中的第一層級，預設值
                WTA_K=2,               # BRIEF 描述符的點數，預設值
                scoreType=cv.ORB_HARRIS_SCORE,  # 特徵點評分方式，預設值
                patchSize=31,          # BRIEF 描述符的區域大小，預設值
                fastThreshold=20       # FAST 角點檢測的閾值，預設值)
                )
    lk_params = dict( winSize  = (18,18),       
                  maxLevel = 5,            
                  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.01))  
    circle_img_range=9
    csv_filename=''
    video_path = select_file("選擇影片檔案", [("Video files", "*.mp4;*.avi;*.mov;*.mkv")])
    print(f'輸入影片：{video_path}')
    if not video_path:
        print("未選擇影片檔案")
        exit()                                
    cap= cv.VideoCapture(video_path)
     #獲取影片輸入參數
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))    
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) 
    fourcc = cv.VideoWriter_fourcc(*'mp4v')          
    fps=cap.get(cv.CAP_PROP_FPS)
    save_path = select_save_path( '保存影片',[("MP4 files", "*.mp4")])
    print(f'輸出影片：{save_path}')
    if save_path:
    # 如果选择了保存路径，初始化 VideoWriter
            out = cv.VideoWriter(save_path, fourcc,fps, (width, height))
    else:
        print("未儲存影片檔案")
    # 调用函数创建 GUI 并获取返回值
    result = create_gui()
    # 处理返回的数据
    if result:
        csv_filename, fan_status, date, fan_speed, video_number = result
        print(f"數據文件: {csv_filename}")
        print(f"風扇狀態: {fan_status}")
        print(f"拍攝日期: {date}")
        print(f"風扇速度: {fan_speed}")
        print(f"影片編號: {video_number}")
    else:
        print("取消文件選擇，不保存數據。")
    # 调用函数创建 GUI 并获取返回值
    feature_detector_type, find_circle = create_selection_gui()
    ret, first_frame = cap.read()
    if not ret:
        print("無法讀取影片")
        exit()
    frame_gray=img_preprocess(first_frame)
    img=frame_gray.copy()
    resized_img,scaling_factor =resize_image_for_screen(img, screen_width, screen_height)
    
    cv.namedWindow('image')
    cv.setMouseCallback('image',on_mouse)
    cv.imshow('image',resized_img)
    print('按Enter確認框選區域，按esc執行光流')
    while True:
        key = cv.waitKey(1)
        if key == 13:  # Enter key
            if not rectangles or rectangles[len(cut_images)-1][0]!=point1 :
                cv.rectangle(resized_img, point1, point2, (0, 0, 255), 1)
                cv.imshow('image',resized_img)
                process_rectangles(scaling_factor)
                print(f"已框選ROI數量: {len(cut_images)}")
                rectangles.append((point1, point2))
            else:
                print('請框選新ROI')
        elif key == 27:  # Esc key
            break 
    start=time.process_time()
    detector = initialize_detector(feature_detector_type, sift_params, surf_params,orb_params)
    p0 = np.zeros((0,1, 2))
    circle_img_list=[]
    if csv_filename:
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Frame','ROI number', 'Point', 'X Coordinate', 'Y Coordinate','fixed X Coordinate','fixed Y Coordinate','Radius','Center X','Center Y','fixed Center X','fixed Center Y','fps','Width','Height','Status','Date','Fan Speed','Video Number','Video Path'])
    for i in range(len(cut_images)):
        cv.imshow('cut_img',cut_images[i])
        kp = detect_keypoints(detector, cut_images[i],feature_detector_type)
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
    circle_img_arry=[]
    for i,old in enumerate(p0):
        a,b = old.ravel()
        a_1,b_1=a,b
        a,b=int(a),int(b)
        
        x, y, radius,combined_img=find_circle_process(find_circle,frame_gray,circle_img_range,a,b)
        if combined_img is not None:
            cv.circle(combined_img, (int(x),int(y)), int(radius), (0, 255, 0), 1)
            circle_img_arry.append(combined_img)
        if csv_filename:
            with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([1,circle_img_number[i],point_list[i], a_1,b_1,a-circle_img_range+x, b-circle_img_range+y, radius,'','','','','','','','','','','',''])
# Create some random colors(繪製光流軌跡線與點用)
    end_surf=time.process_time()
    surf_time = end_surf-start
    print(f'特徵檢索時間：{surf_time}s')
    color=np.zeros((len(p0),3))
    old_gray=frame_gray
# Create a mask image for drawing purposes
    mask = np.zeros_like(first_frame)
    mask[mask==0] ='255'
    frame_number=1
    total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))-1
    with tqdm(total=total_frames, unit='frame') as pbar:
        while(1):
            ret,frame = cap.read()
        #如果讀不到圖片則退出
            if frame is None:
                break
            frame_number=frame_number+1
        #將影像轉為灰度圖片
            #first_gray = second_gray
            second_gray = img_preprocess(frame)
            frame_gray = second_gray
        # calculate optical flow
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray,p0, None, **lk_params) 
        # draw the tracks(i個角點)
            circle_img_arry=[]
            for i,(new,old) in enumerate(zip(p1, p0)):
                a,b = new.ravel() # 新點
                a_1,b_1=a,b
                c,d = old.ravel() # 舊點
                a, b, c, d = int(a), int(b), int(c), int(d)
                if st[i]==1:
                    color[i]=(0,255,0)
                else:
                    color[i]=(0,0,255)
                x, y, radius,combined_img=find_circle_process(find_circle,frame_gray,circle_img_range,a,b)
                if combined_img is not None:
                    resized_img = cv.resize(combined_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv.INTER_LINEAR)
                    cv.circle(resized_img, (int(x*scale_factor),int(y*scale_factor)), int(radius*scale_factor), (0, 255, 0), 1)
                    circle_img_arry.append(resized_img)
                if csv_filename:
                    with open(csv_filename, mode='a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([frame_number,circle_img_number[i], point_list[i], a_1,b_1,a-circle_img_range+x, b-circle_img_range+y, radius,'','','','','','','','','','','',''])
                frame = cv.circle(frame,(a,b),4,color[i].tolist(),-1)
                winSize=lk_params['winSize']
                win_x=int(winSize[0]/2)
                win_y=int(winSize[1]/2)
                tracks = cv.line(mask, (a,b),(c,d), color[i].tolist(), 1)
                #frame = cv.circle(frame,(int(a-circle_img_range+x),int(b-circle_img_range+y)), radius_int, (0, 255, 0), 1)
                frame = cv.putText(frame,'Point'+str(i),
                        (a,b), cv.FONT_HERSHEY_SIMPLEX,0.5, color[i].tolist(),1, cv.LINE_AA)
                frame = cv.rectangle(frame, (c-win_x, d-win_y), (c+win_x,d+win_y), color[i].tolist(), 1) 
                
            if find_circle:
                combined_circle_img = np.hstack(circle_img_arry)
                if combined_circle_img.shape[1] > 0:  # 如果 ROI 的宽度和高度都大于 0
                        cv.imshow('combined_circle_img', combined_circle_img)
            frame = cv.putText(frame,'SURF point: '+str(i+1)+'   '+'SURF Parameters '+str(list(surf_params.items())),
                        (10,20), cv.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0), 1, cv.LINE_AA)
            frame = cv.putText(frame,'LK Parameters '+str(list(lk_params.items())),
                        (10,40), cv.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0), 1, cv.LINE_AA)
            frame = cv.putText(frame,'Frame:'+str(frame_number)+' fps:'+str(fps),
                        (10,60), cv.FONT_HERSHEY_SIMPLEX,0.5, (0,0,0), 1, cv.LINE_AA) 
            tracks_gray = cv.cvtColor(tracks, cv.COLOR_BGR2GRAY)  
            ret, mask1  = cv.threshold(tracks_gray, 200, 255, cv.THRESH_BINARY_INV)
            path = cv.bitwise_and(tracks,tracks, mask = mask1 )
            ret, mask2  = cv.threshold(tracks_gray, 200, 255, cv.THRESH_BINARY)
            bg1 = cv.bitwise_and(frame, frame, mask = mask2 )
            frame_gray1=np.stack((frame_gray,) * 3, axis=-1)

            bg2 = cv.bitwise_and(frame_gray1, frame_gray1, mask = mask2 )
            img1 = cv.add(bg1,path)
            img2 = cv.add(bg2,path)
            crop_x = 1000
            crop_y = 1000
            resized_img1 =cv.resize(img1, (img1.shape[1] // 2, img1.shape[0] // 2))
            #resized_img1 = crop_center(resized_img1, crop_x, crop_y)
            combined_img1 = np.concatenate((img1,img2), axis=1)
            resized_img2 =cv.resize(combined_img1, (combined_img1.shape[1] // 4, combined_img1.shape[0] // 4))
            if save_path:
                    out.write(img1)
            cv.imshow('frame',resized_img1)
        #按esc鍵停止迴圈 
            k = cv.waitKey(70) & 0xff
            if k == 27:
                break
        # Now update the previous frame and previous points
            old_gray = frame_gray.copy()
            p0 = p1.reshape(-1,1,2)
            pbar.update(1)
    end_optical_flow=time.process_time()
    print(f'光流花費時間：{end_optical_flow-end_surf}s')
    if save_path:
            out.release()
    cap.release()
    cv.destroyAllWindows() 
    if csv_filename:  
        with open(csv_filename, mode='r', newline='') as file:
            reader = csv.reader(file)
            rows = list(reader)
            rows[1][12:20] = [fps,width,height,fan_status,date,fan_speed,video_number,video_path]
        with open(csv_filename, mode='w', newline='',encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
        print("CSV file saved successfully:", csv_filename)
    
if __name__=='__main__':
    main()
