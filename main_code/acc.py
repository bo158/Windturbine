import pandas as pd
import numpy as np
from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
import os

# 初始化 Tkinter 根窗口
root = Tk()
root.withdraw()  # 隐藏根窗口

# 打开文件对话框，选择 CSV 文件
file_path = askopenfilename(
    filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
)

# 检查是否选择了文件
if file_path:
    if os.path.exists(file_path):
        try:
            # 使用 BIG5 编码读取文件
            df = pd.read_csv(file_path, encoding='big5')

            # 确保文件中包含所需的列
            if all(col in df.columns for col in ['AccX[G]', 'AccY[G]', 'AccZ[G]']):
                # 提取 AccX, AccY 和 AccZ 列
                acc_x = df['AccX[G]'].values
                acc_y = df['AccY[G]'].values
                acc_z = df['AccZ[G]'].values

                # 将三个数列组合成一个形状为 [n, 3] 的 NumPy 数组
                acc_data = np.column_stack((acc_x, acc_y, acc_z))

                # 打开保存文件对话框
                save_path = asksaveasfilename(
                    defaultextension=".npy",
                    filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")]
                )

                # 检查是否选择了保存位置
                if save_path:
                    # 保存 NumPy 数组到 .npy 文件
                    np.save(save_path, acc_data)
                    print(f"数据已成功保存到 {save_path}")
                else:
                    print("未选择保存位置")
            else:
                print("CSV 文件中缺少所需的列")
        except Exception as e:
            print(f"读取文件时出错: {e}")
    else:
        print("文件路径不存在")
else:
    print("未选择文件")
