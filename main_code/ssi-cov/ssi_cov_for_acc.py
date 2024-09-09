import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyoma2.algorithm import FDD_algo, FSDD_algo, SSIcov_algo
from pyoma2.OMA import SingleSetup
import tkinter as tk
from tkinter.filedialog import askopenfilename

# 使用 GUI 選擇檔案的函式
def select_file(title, filetypes):
    root = tk.Tk()
    root.withdraw()  # 隱藏主視窗
    try:
        filepath = askopenfilename(title=title, filetypes=filetypes)
    finally:
        root.destroy()  # 確保在選擇檔案後銷毀 Tkinter 資源
    return filepath

# Add the directory we executed the script from to path:
sys.path.insert(0, os.path.realpath('__file__'))

# Load example dataset for single setup
# 提示使用者選擇 .npy 檔案
npy_filename = select_file("請選擇 .npy 檔案", [("Numpy 檔案", "*.npy")])

# 檢查是否選擇了檔案
if npy_filename:
    # 載入 .npy 檔案
    data = np.load(npy_filename, allow_pickle=True)
    print(f"數據成功從檔案載入: {npy_filename}")
    # 現在可以使用 `data` 進行進一步處理
else:
    print("未選擇檔案或選擇了不支持的檔案類型")
#/pyOMA2-0.5.2/pyOMA2-0.5.2/src/pyoma2/test_data/palisaden/Palisaden_dataset.npy

# Create single setup
Pali_ss = SingleSetup(data, fs=100)
# Detrend and decimate
Pali_ss.filter_data(Wn= [5 / 50, 20 / 50] , order= 4, btype = 'bandpass') 
print(Pali_ss.fs)
#Pali_ss.detrend_data()
Pali_ss.decimate_data(q=2, inplace=True) # q=decimation factor
print(Pali_ss.fs)


# Enable interactive mode for matplotlib
plt.ion()

# Plot the Time Histories
fig1, ax1 = Pali_ss.plot_data()
#plt.show(block=True)

# Plot TH, PSD and KDE of the (selected) channels
fig2, ax2 = Pali_ss.plot_ch_info(ch_idx=[0])
#plt.show(block=True)
# Initialise the algorithms
fdd = FDD_algo(name="FDD")
fsdd = FSDD_algo(name="FSDD", nxseg=1024, method_SD="per", pov=0.5)
ssicov = SSIcov_algo(name="SSIcov", br=200, ordmax=60)

# Overwrite/update run parameters for an algorithm
fdd.run_params = FDD_algo.RunParamCls(nxseg=512, method_SD="cor")
# Aggiungere esempio anche col metodo

# Add algorithms to the single setup class
Pali_ss.add_algorithms(ssicov, fsdd, fdd)

# Run all or run by name
Pali_ss.run_by_name("SSIcov")
Pali_ss.run_by_name("FSDD")
# Pali_ss.run_all()

# save dict of results
ssi_res = ssicov.result.model_dump()
fsdd_res = dict(fsdd.result)

# plot Stabilisation chart for SSI
# plot Singular values of PSD
fig3, ax3 = fsdd.plot_CMIF(freqlim=(5,20))
plt.show(block=True)
fig4, ax4 = ssicov.plot_STDiag(freqlim=(5,20), hide_poles=False)
plt.show(block=True)
# plot frequecy-damping clusters for SSI
fig5, ax5 = ssicov.plot_cluster(freqlim=(5,20))
plt.show(block=True)

# Select modes to extract from plots
Pali_ss.MPE_fromPlot("SSIcov", freqlim=(5,20))

# or directly
#Pali_ss.MPE("SSIcov", sel_freq=[1.88, 2.42, 2.68], order=40)

# update dict of results
ssi_res = dict(ssicov.result)

# Select modes to extract from plots
Pali_ss.MPE_fromPlot("FSDD", freqlim=(5,20), MAClim=0.95)

# or directly
#Pali_ss.mpe("FSDD", sel_freq=[1.88, 2.42, 2.68], MAClim=0.95)

# update dict of results
fsdd_res = dict(fsdd.result)

# print the results
print(f"ssi-data\n")
print(f"order out: {ssi_res['order_out']} \n")
print(f"the natural frequencies are: {ssi_res['Fn']} \n")

print(f"fsdd-data\n")
print(f"the natural frequencies are: {fsdd_res['Fn']} \n")