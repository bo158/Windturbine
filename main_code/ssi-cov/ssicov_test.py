import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pyoma2.algorithm import FDD_algo, FSDD_algo, SSIcov_algo
from pyoma2.OMA import SingleSetup
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class OMAApp:
    def __init__(self, root):
        self.root = root
        self.root.title("OMA 分析工具")
        
        # 在視窗關閉時觸發退出程序
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 檔案選擇
        self.file_label = ttk.Label(root, text="選擇 Numpy 檔案:")
        self.file_label.grid(row=0, column=0, padx=10, pady=10)
        self.file_button = ttk.Button(root, text="選擇檔案", command=self.select_file)
        self.file_button.grid(row=0, column=1, padx=10, pady=10)
        
        self.filepath_label = ttk.Label(root, text="未選擇檔案")  # Label to display selected file path
        self.filepath_label.grid(row=0, column=2, padx=10, pady=10, columnspan=2, sticky="w")

        self.filepath = None
        
        # 取樣頻率
        self.fs_label = ttk.Label(root, text="取樣頻率 (Hz):")
        self.fs_label.grid(row=1, column=0, padx=10, pady=10)
        self.fs_entry = ttk.Entry(root)
        self.fs_entry.grid(row=1, column=1, padx=10, pady=10)
        
        # 濾波器選項
        self.filter_label = ttk.Label(root, text="高通 / 低通頻率 (Hz):")
        self.filter_label.grid(row=2, column=0, padx=10, pady=10)
        self.filter_hp_entry = ttk.Entry(root)
        self.filter_hp_entry.grid(row=2, column=1, padx=10, pady=10)
        self.filter_lp_entry = ttk.Entry(root)
        self.filter_lp_entry.grid(row=2, column=2, padx=10, pady=10)
        
        # 濾波器類型選擇
        self.filter_type_label = ttk.Label(root, text="濾波器類型:")
        self.filter_type_label.grid(row=3, column=0, padx=10, pady=10)
        self.filter_type_combobox = ttk.Combobox(root, values=["不使用", "低通", "高通", "帶通"])
        self.filter_type_combobox.grid(row=3, column=1, padx=10, pady=10)
        self.filter_type_combobox.current(0)  # 預設選擇 "不使用"
        
        # 是否使用抽取
        self.decimate_var = tk.IntVar()
        self.decimate_check = ttk.Checkbutton(root, text="使用抽取", variable=self.decimate_var, command=self.toggle_decimate)
        self.decimate_check.grid(row=4, column=0, padx=10, pady=10)
        
        self.decimate_label = ttk.Label(root, text="抽取因子:")
        self.decimate_label.grid(row=4, column=1, padx=10, pady=10)
        self.decimate_entry = ttk.Entry(root)
        self.decimate_entry.grid(row=4, column=2, padx=10, pady=10)
        self.decimate_entry.config(state=tk.DISABLED)
        
        # Block row 和 Order
        self.br_label = ttk.Label(root, text="Block Row:")
        self.br_label.grid(row=5, column=0, padx=10, pady=10)
        self.br_entry = ttk.Entry(root)
        self.br_entry.grid(row=5, column=1, padx=10, pady=10)
        
        self.order_label = ttk.Label(root, text="Order:")
        self.order_label.grid(row=5, column=2, padx=10, pady=10)
        self.order_entry = ttk.Entry(root)
        self.order_entry.grid(row=5, column=3, padx=10, pady=10)
        
        # SSI-COV 頻率範圍
        self.ssi_freq_label = ttk.Label(root, text="SSI-COV 頻率範圍 (Hz):")
        self.ssi_freq_label.grid(row=6, column=0, padx=10, pady=10)
        self.ssi_freq_entry_min = ttk.Entry(root)
        self.ssi_freq_entry_min.grid(row=6, column=1, padx=10, pady=10)
        self.ssi_freq_entry_min.insert(0, "1")  # 預設最小值
        self.ssi_freq_entry_max = ttk.Entry(root)
        self.ssi_freq_entry_max.grid(row=6, column=2, padx=10, pady=10)
        self.ssi_freq_entry_max.insert(0, "15")  # 預設最大值
        
        # 提交按鈕
        self.submit_button = ttk.Button(root, text="開始分析", command=self.start_analysis)
        self.submit_button.grid(row=7, column=0, columnspan=4, padx=10, pady=10)
        
        # Matplotlib 圖形顯示區域
        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().grid(row=8, column=0, columnspan=4, padx=10, pady=10)

        # Bind resize event
        self.root.bind("<Configure>", self.on_resize)
    
    def toggle_decimate(self):
        if self.decimate_var.get():
            self.decimate_entry.config(state=tk.NORMAL)
        else:
            self.decimate_entry.config(state=tk.DISABLED)
    
    def select_file(self):
        self.filepath = filedialog.askopenfilename(title="請選擇 .npy 檔案", filetypes=[("Numpy 檔案", "*.npy")])
        if self.filepath:
            self.filepath_label.config(text=self.filepath)  # Update label with the selected file path
            messagebox.showinfo("檔案選擇", f"已選擇檔案: {self.filepath}")
    
    def start_analysis(self):
        # 取得參數
        fs = float(self.fs_entry.get())
        hp_freq = float(self.filter_hp_entry.get()) if self.filter_hp_entry.get() else None
        lp_freq = float(self.filter_lp_entry.get()) if self.filter_lp_entry.get() else None
        br = int(self.br_entry.get())
        order = int(self.order_entry.get())
        
        if not self.filepath:
            messagebox.showerror("錯誤", "請選擇檔案")
            return
        
        # 載入數據
        data = np.load(self.filepath, allow_pickle=True)
        
        # 創建 SingleSetup
        setup = SingleSetup(data, fs=fs)
        
        # 濾波器類型選擇
        filter_type = self.filter_type_combobox.get()
        if filter_type == "低通":
            if lp_freq is None:
                messagebox.showerror("錯誤", "請輸入低通頻率")
                return
            setup.filter_data(Wn=lp_freq, order=8, btype='lowpass')
        elif filter_type == "高通":
            if hp_freq is None:
                messagebox.showerror("錯誤", "請輸入高通頻率")
                return
            setup.filter_data(Wn=hp_freq, order=8, btype='highpass')
        elif filter_type == "帶通":
            if hp_freq is None or lp_freq is None:
                messagebox.showerror("錯誤", "請輸入高通和低通頻率")
                return
            setup.filter_data(Wn=[hp_freq, lp_freq], order=8, btype='bandpass')
        
        # 是否使用抽取
        if self.decimate_var.get():
            try:
                q = int(self.decimate_entry.get())
                if q < 1:
                    raise ValueError("抽取因子必須大於0")
                setup.decimate_data(q=q, inplace=True)
            except ValueError as e:
                messagebox.showerror("錯誤", str(e))
                return
        
        # 初始化算法
        ssicov = SSIcov_algo(name="SSIcov", br=br, ordmax=order)
        setup.add_algorithms(ssicov)
        setup.run_by_name("SSIcov")
        
        # 取得結果並繪圖
        self.plot_results(ssicov)
    
    def plot_results(self, ssicov):
        # 清理现有图形
        self.ax.clear()

        # 确保图形绘制到正确的 figure
        plt.figure(self.figure.number)

        # 取得 SSI-COV 頻率範圍
        freq_min = float(self.ssi_freq_entry_min.get())
        freq_max = float(self.ssi_freq_entry_max.get())

        # 繪製穩定圖
        fig5, ax5 = ssicov.plot_STDiag(freqlim=(freq_min, freq_max), hide_poles=True)

        # 更新 canvas 尺寸
        self.figure.set_size_inches(self.canvas.get_tk_widget().winfo_width() / self.figure.dpi,
                                    self.canvas.get_tk_widget().winfo_height() / self.figure.dpi)

        # 更新 canvas 和 figure
        self.canvas.figure = fig5
        self.canvas.draw()

        # 強制刷新 Matplotlib 圖像
        self.root.update_idletasks()
    
    def on_resize(self, event):
        """處理窗口大小變化事件"""
        if self.canvas is not None:
            # 重新配置 canvas 尺寸
            self.figure.set_size_inches(self.canvas.get_tk_widget().winfo_width() / self.figure.dpi,
                                        self.canvas.get_tk_widget().winfo_height() / self.figure.dpi)
            self.canvas.draw()

    def on_closing(self):
        """當視窗關閉時結束程序"""
        if messagebox.askokcancel("退出", "你確定要退出程式嗎?"):
            self.root.quit()  # 停止Tkinter主循環
            self.root.destroy()  # 正常關閉視窗
            sys.exit()  # 終止整個程式

if __name__ == "__main__":
    root = tk.Tk()
    app = OMAApp(root)
    root.mainloop()
