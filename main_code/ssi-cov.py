import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyoma2.algorithm import FDD_algo, FSDD_algo, SSIcov_algo
from pyoma2.OMA import SingleSetup

# Add the directory we executed the script from to path:
sys.path.insert(0, os.path.realpath('__file__'))

# Load example dataset for single setup
data = np.load("D:/dfg22/Downloads/data.npy", allow_pickle=True)
#/pyOMA2-0.5.2/pyOMA2-0.5.2/src/pyoma2/test_data/palisaden/Palisaden_dataset.npy

# Create single setup
Pali_ss = SingleSetup(data, fs=350)
# Detrend and decimate
#Pali_ss.detrend_data()
Pali_ss.decimate_data(q=4, inplace=True) # q=decimation factor
print(Pali_ss.fs)


# Enable interactive mode for matplotlib
plt.ion()

# Plot the Time Histories
fig1, ax1 = Pali_ss.plot_data()
plt.show(block=True)

# Plot TH, PSD and KDE of the (selected) channels
fig2, ax2 = Pali_ss.plot_ch_info(ch_idx=[0])
plt.show(block=True)
# Initialise the algorithms
fdd = FDD_algo(name="FDD")
fsdd = FSDD_algo(name="FSDD", nxseg=2048, method_SD="per", pov=0.5)
ssicov = SSIcov_algo(name="SSIcov", br=50, ordmax=80)

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
fig3, ax3 = fsdd.plot_CMIF(freqlim=(0.1,30))
plt.show(block=True)
fig4, ax4 = ssicov.plot_STDiag(freqlim=(0.1,30), hide_poles=False)
plt.show(block=True)
# plot frequecy-damping clusters for SSI
fig5, ax5 = ssicov.plot_cluster(freqlim=(0.1,30))
plt.show(block=True)

# Select modes to extract from plots
Pali_ss.MPE_fromPlot("SSIcov", freqlim=(0.1,30))

# or directly
#Pali_ss.MPE("SSIcov", sel_freq=[1.88, 2.42, 2.68], order=40)

# update dict of results
ssi_res = dict(ssicov.result)
print(ssicov.result.Fn)