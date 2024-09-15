import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pyoma2.algorithm import SSIcov_algo
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
        
        # 框架
        self.frame = ttk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 檔案選擇
        self.file_frame = ttk.Frame(self.frame)
        self.file_frame.pack(fill=tk.X)
        
        self.file_label = ttk.Label(self.file_frame, text="選擇 Numpy 檔案:")
        self.file_label.pack(side=tk.LEFT)
        
        self.file_button = ttk.Button(self.file_frame, text="選擇檔案", command=self.select_file)
        self.file_button.pack(side=tk.LEFT, padx=5)
        
        self.filepath_label = ttk.Label(self.file_frame, text="未選擇檔案")
        self.filepath_label.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        self.filepath = None
        
        # 取樣頻率
        self.fs_frame = ttk.Frame(self.frame)
        self.fs_frame.pack(fill=tk.X)
        
        self.fs_label = ttk.Label(self.fs_frame, text="取樣頻率 (Hz):")
        self.fs_label.pack(side=tk.LEFT)
        
        self.fs_entry = ttk.Entry(self.fs_frame)
        self.fs_entry.pack(side=tk.LEFT, padx=5)
        self.fs_entry.insert(0, "100")
        
        # 濾波器選項
        self.filter_frame = ttk.Frame(self.frame)
        self.filter_frame.pack(fill=tk.X)
        
        self.filter_label = ttk.Label(self.filter_frame, text="高通 / 低通頻率 (Hz):")
        self.filter_label.pack(side=tk.LEFT)
        
        self.filter_hp_entry = ttk.Entry(self.filter_frame)
        self.filter_hp_entry.pack(side=tk.LEFT, padx=5)
        
        self.filter_lp_entry = ttk.Entry(self.filter_frame)
        self.filter_lp_entry.pack(side=tk.LEFT, padx=5)
        
        # 濾波器類型選擇
        self.filter_type_frame = ttk.Frame(self.frame)
        self.filter_type_frame.pack(fill=tk.X)
        
        self.filter_type_label = ttk.Label(self.filter_type_frame, text="濾波器類型:")
        self.filter_type_label.pack(side=tk.LEFT)
        
        self.filter_type_combobox = ttk.Combobox(self.filter_type_frame, values=["不使用", "低通", "高通", "帶通"])
        self.filter_type_combobox.pack(side=tk.LEFT, padx=5)
        self.filter_type_combobox.current(0)
        
        # 是否使用抽取
        self.decimate_frame = ttk.Frame(self.frame)
        self.decimate_frame.pack(fill=tk.X)
        
        self.decimate_var = tk.IntVar()
        self.decimate_check = ttk.Checkbutton(self.decimate_frame, text="使用抽取", variable=self.decimate_var, command=self.toggle_decimate)
        self.decimate_check.pack(side=tk.LEFT)
        
        self.decimate_label = ttk.Label(self.decimate_frame, text="抽取因子:")
        self.decimate_label.pack(side=tk.LEFT, padx=5)
        
        self.decimate_entry = ttk.Entry(self.decimate_frame)
        self.decimate_entry.pack(side=tk.LEFT, padx=5)
        self.decimate_entry.config(state=tk.DISABLED)
        
        # Block row 和 Order
        self.block_order_frame = ttk.Frame(self.frame)
        self.block_order_frame.pack(fill=tk.X)
        
        self.br_label = ttk.Label(self.block_order_frame, text="Block Row:")
        self.br_label.pack(side=tk.LEFT)
        
        self.br_entry = ttk.Entry(self.block_order_frame)
        self.br_entry.pack(side=tk.LEFT, padx=5)
        self.br_entry.insert(0, "80")
        
        self.order_label = ttk.Label(self.block_order_frame, text="Order:")
        self.order_label.pack(side=tk.LEFT, padx=5)
        
        self.order_entry = ttk.Entry(self.block_order_frame)
        self.order_entry.pack(side=tk.LEFT, padx=5)
        self.order_entry.insert(0, "50")

        # SSI-COV 頻率範圍
        self.ssi_freq_frame = ttk.Frame(self.frame)
        self.ssi_freq_frame.pack(fill=tk.X)
        
        self.ssi_freq_label = ttk.Label(self.ssi_freq_frame, text="SSI-COV 頻率範圍 (Hz):")
        self.ssi_freq_label.pack(side=tk.LEFT)
        
        self.ssi_freq_entry_min = ttk.Entry(self.ssi_freq_frame)
        self.ssi_freq_entry_min.pack(side=tk.LEFT, padx=5)
        self.ssi_freq_entry_min.insert(0, "1")
        
        self.ssi_freq_entry_max = ttk.Entry(self.ssi_freq_frame)
        self.ssi_freq_entry_max.pack(side=tk.LEFT, padx=5)
        self.ssi_freq_entry_max.insert(0, "15")
        
        # Hide Poles Checkbox
        self.hide_poles_var = tk.IntVar()
        self.hide_poles_check = ttk.Checkbutton(self.frame, text="隱藏不穩點", variable=self.hide_poles_var)
        self.hide_poles_check.pack(pady=5)

        # 顯示自然頻率的選項
        self.show_freq_var = tk.IntVar()
        self.show_freq_check = ttk.Checkbutton(self.frame, text="提取自然頻率", variable=self.show_freq_var)
        self.show_freq_check.pack(pady=5)

        # 提交按鈕
        self.submit_button = ttk.Button(self.frame, text="開始分析", command=self.start_analysis)
        self.submit_button.pack(pady=10)

        # 顯示自然頻率的區域
        self.freq_label = ttk.Label(self.frame, text="", wraplength=400)
        self.freq_label.pack(pady=5)

    def toggle_decimate(self):
        if self.decimate_var.get():
            self.decimate_entry.config(state=tk.NORMAL)
        else:
            self.decimate_entry.config(state=tk.DISABLED)
    
    def select_file(self):
        self.filepath = filedialog.askopenfilename(title="請選擇 .npy 檔案", filetypes=[("Numpy 檔案", "*.npy")])
        if self.filepath:
            self.filepath_label.config(text=self.filepath)  # Update label with the selected file path
    
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
        self.plot_results(ssicov,setup)


    def plot_results(self, ssicov,setup):
        try:
            freq_min = float(self.ssi_freq_entry_min.get())
            freq_max = float(self.ssi_freq_entry_max.get())
        except ValueError:
            messagebox.showerror("錯誤", "請輸入有效的頻率範圍")
            return
        hide_poles = self.hide_poles_var.get() == 1  # Check if "隱藏極點" is selected

        fig5, ax5 = ssicov.plot_STDiag(freqlim=(freq_min, freq_max), hide_poles=hide_poles)
        if self.show_freq_var.get():
            freq_min = float(self.ssi_freq_entry_min.get())
            freq_max = float(self.ssi_freq_entry_max.get())
            setup.MPE_fromPlot("SSIcov", freqlim=(freq_min, freq_max))
            ssi_res = dict(ssicov.result)
            # Get the natural frequencies
            frequencies = ssi_res.get('Fn', [])
            # Round frequencies to the nearest integer or specified decimal places
            frequencies = [round(freq, 2) for freq in frequencies]  # Adjust decimal places as needed
            freq_text = "自然頻率: " + ', '.join(f"{freq:.2f} Hz" for freq in frequencies)
            self.freq_label.config(text=freq_text)

            # Plot vertical lines and labels for each natural frequency
            for freq in frequencies:
                ax5.axvline(x=freq, color='r', linestyle='--', linewidth=1)
                ax5.text(freq, 0.5, f'{freq:.2f}', color='r', verticalalignment='center', horizontalalignment='right', fontsize=12, transform=ax5.get_xaxis_transform())
        else:
            self.freq_label.config(text='')  
        # 清除先前的圖形
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()  # 銷毀圖形
            del self.canvas  # 確保引用被刪除

        # 新建圖形並顯示
        self.canvas = FigureCanvasTkAgg(fig5, master=self.frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.root.update_idletasks()
        self.root.update()        
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
