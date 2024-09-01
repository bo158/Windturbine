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
cut_rectangles = []
cut_images = []
min_x_values = []
min_y_values = []
def get_screen_size():
    root = tk.Tk()
    root.withdraw()  # 隱藏主窗口
    screen_width = root.winfo_screenwidth() - 100
    screen_height = root.winfo_screenheight() - 100
    root.destroy()  # 銷毀主窗口
    return screen_width, screen_height
def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    try:
        # 打开文件对话框
        filepath = askopenfilename(title=title, filetypes=filetypes)
    finally:
        # 确保 Tkinter 资源被销毁
        root.destroy()
    return filepath

def select_save_path(file_types):
    # 创建 Tkinter 主窗口并隐藏
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    # 打开保存文件对话框
    try:
        filepath = filedialog.asksaveasfilename(title="保存文件",filetypes=file_types)
    finally:
        # 确保 Tkinter 资源被销毁
        root.destroy()
    return filepath

def create_selection_gui():
    def submit_selection():
        global feature_detector_type
        # 获取用户选择的特征检测器类型
        feature_detector_type = feature_detector_var.get()
        
        
        # 打印选项到控制台（可选）
        print(f"选择的特征检测器: {feature_detector_type}")

        # 关闭窗口
        root.destroy()

    # 创建主窗口
    global root, feature_detector_var
    root = tk.Tk()
    root.title("选择特征检测器和目标类型")

    # 创建特征检测器选择
    tk.Label(root, text="选择特征检测器:").pack(padx=10, pady=10)
    feature_detector_var = tk.StringVar(value="sift")
    detectors = ["sift", "surf", "orb", "shi"]
    for detector in detectors:
        tk.Radiobutton(root, text=detector.upper(), variable=feature_detector_var, value=detector).pack(anchor=tk.W)

    # 创建提交按钮
    submit_button = tk.Button(root, text="提交", command=submit_selection)
    submit_button.pack(pady=20)

    # 运行主循环
    root.mainloop()

    return feature_detector_type
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
    global point1, point2
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
def process_rectangle(scaling_factor,point3,point4):
    min_x = min(point3[0], point4[0]) * (1 / scaling_factor)
    min_y = min(point3[1], point4[1]) * (1 / scaling_factor)
    width = abs(point3[0] - point4[0]) * (1 / scaling_factor)
    height = abs(point3[1] - point4[1]) * (1 / scaling_factor)

    min_x = int(min_x)
    min_y = int(min_y)
    width = int(width)
    height = int(height)

    center_x = min_x + width // 2
    center_y = min_y + height // 2
   
    return center_x, center_y, width, height
   
def cut_frame(frame_for_cut,x,y,width,height):
    img_height, img_width, _ = frame_for_cut.shape

    # 計算裁切矩形的邊界
    left = x - width // 2
    top = y - height // 2
    right = x + width // 2
    bottom = y + height // 2
    
    # 計算裁切矩形是否超出原圖片範圍
    new_left = max(left, 0)
    new_top = max(top, 0)
    new_right = min(right, img_width)
    new_bottom = min(bottom, img_height)

    # 創建一個新的圖片（填充為黑色）
    new_img_width = max(width, new_right - new_left)
    new_img_height = max(height, new_bottom - new_top)
    new_image = np.zeros((new_img_height, new_img_width, 3), dtype=np.uint8)
   
    # 裁切原圖片
    cropped_image =frame_for_cut[new_top:new_bottom, new_left:new_right]

    # 計算裁切圖片在新圖片中的位置
    x_offset = new_left - left
    y_offset = new_top - top

    # 粘貼裁切圖片到新圖片上
    new_image[y_offset:y_offset + cropped_image.shape[0], x_offset:x_offset + cropped_image.shape[1]] = cropped_image

    return new_image

    

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
    image = cv.GaussianBlur(img, (3, 3), 0)
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    #image = cv.bilateralFilter(image, d=5, sigmaColor=10, sigmaSpace=10)
    #image = cv.medianBlur(image,5)
    #image = cv.Canny(image, 170, 170)
    #thre, image= cv.threshold(image, 125, 255, cv.THRESH_BINARY)
    return image

def main():
    #設定參數
    global resized_img,first_frame,version
    version = cv.__version__

    screen_width, screen_height = get_screen_size()

    sift_params = dict(nfeatures = 10,
                   nOctaveLayers = 3,
                   contrastThreshold = 0.04,
                   edgeThreshold = 20,
                   sigma =1.6 )
    surf_params = dict(hessianThreshold= 100,
                   nOctaves= 3,
                   nOctaveLayers =3)
    orb_params = dict(nfeatures = 10,  
                        nlevels = 4, 
                        edgeThreshold = 7)
    lk_params = dict( winSize  = (12,12),       
                  maxLevel = 5,            
                  criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.01))  
    circle_img_range=10

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
    save_path = select_save_path( [("MP4 files", "*.mp4")])
    print(f'輸出影片：{save_path}')
    
   
    feature_detector_type= create_selection_gui()
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
  
    
    cv.imshow('cut_img',cut_images[0])
    kp = detect_keypoints(detector, cut_images[0],feature_detector_type)
    kp[:, 0] += min_x_values[0]
    kp[:, 1] += min_y_values[0]
    kp2 = kp.reshape(-1, 1, 2) 
    coordinates_array = np.array(kp2)
    kp2 = np.unique(coordinates_array, axis=0)
    circle_img_list.append(len(kp2))
    p0 = np.concatenate((p0, kp2), axis=0)
    arrays = [np.arange(1, count + 1) for count in circle_img_list]
    point_list = np.concatenate(arrays)
    p0 = p0.astype('float32')
    for i,old in enumerate(p0):
        a,b = old.ravel()
        a_1,b_1=a,b
        a,b=int(a),int(b)        
# Create some random colors(繪製光流軌跡線與點用)
    end_surf=time.process_time()
    surf_time = end_surf-start
    print(f'特徵檢索時間：{surf_time}s')

    resized_img,scaling_factor =resize_image_for_screen(img, screen_width, screen_height)

    cv.namedWindow('image')
    cv.setMouseCallback('image',on_mouse)
    cv.imshow('image',resized_img)
    print('按Enter確認裁剪區域')
    
    while True:
        key = cv.waitKey(1)
        if key==27:
            cv.rectangle(resized_img, point1, point2, (0, 0, 255), 1)
            cv.imshow('image',resized_img)
            print(f"已框選ROI")
            point3=point1
            point4=point2
            break
    center_x, center_y, cut_width, cut_height=process_rectangle(scaling_factor,point3,point4)
    cutted_frame=cut_frame(first_frame,center_x,center_y,cut_width,cut_height)
    center_x_dic=center_x-a_1
    center_y_dic=center_y-b_1
    resize_cutted_frame,_=resize_image_for_screen(cutted_frame, screen_width, screen_height)
    if save_path:
    # 如果选择了保存路径，初始化 VideoWriter
            out = cv.VideoWriter(save_path, fourcc,fps, (cut_width, cut_height))
    else:
        print("未儲存影片檔案")
    cv.imshow('cutted_frame',resize_cutted_frame)
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
            frame_for_cut=frame.copy()
            for i,(new,old) in enumerate(zip(p1, p0)):
                a,b = new.ravel() # 新點
                a_1,b_1=a,b
                c,d = old.ravel() # 舊點
                a_2,b_2,c_2,d_2=a,b,c,d
                a, b, c, d = int(a), int(b), int(c), int(d)
                
                if st[i]==1:
                    color[i]=(0,255,0)
                else:
                    color[i]=(0,0,255)
                frame = cv.circle(frame,(a,b),4,color[i].tolist(),-1)
                winSize=lk_params['winSize']
                win_x=int(winSize[0]/2)
                win_y=int(winSize[1]/2)
                tracks = cv.line(mask, (a,b),(c,d), color[i].tolist(), 1)
                #frame = cv.circle(frame,(int(a-circle_img_range+x),int(b-circle_img_range+y)), radius_int, (0, 255, 0), 1)
                frame = cv.putText(frame,'Point'+str(i),
                        (a,b), cv.FONT_HERSHEY_SIMPLEX,0.5, color[i].tolist(),1, cv.LINE_AA)
                frame = cv.rectangle(frame, (c-win_x, d-win_y), (c+win_x,d+win_y), color[i].tolist(), 1) 
            center_x=int(center_x_dic+c_2)
            center_y=int(center_y_dic+d_2)
            cutted_frame=cut_frame(frame_for_cut,center_x,center_y,cut_width,cut_height)

            resize_cutted_frame,_=resize_image_for_screen(cutted_frame, screen_width, screen_height)
            cv.imshow('cutted_frame',resize_cutted_frame)
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
                    out.write(cutted_frame)
            #cv.imshow('frame',resized_img1)
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
if __name__=='__main__':
    main()