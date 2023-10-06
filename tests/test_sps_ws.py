"""
Test script to properly read and fBCT file from SPS
"""
import matplotlib.pyplot as plt
from ccc_miner import WS

# Load data into fBCT
parquet_file = '../data/test_data/WS_2023.09.22.16.08.02.376100.parquet'

# Test data corresponds to this logbook entry:
# https://logbook.cern.ch/elogbook-server/GET/showEventInLogbook/3835189

# Instantiate class 
ws = WS(parquet_file)

# First test 
figure_X, n_emittances_X, sigmas_raw_X = ws.fit_Gaussian_To_and_Plot_Relevant_Profiles() 
figure_Y, n_emittances_Y, sigmas_raw_Y = ws.fit_Gaussian_To_and_Plot_Relevant_Profiles(plane='Y') 
plt.show()


#fig.savefig('Test_plot.png', dpi=250)
