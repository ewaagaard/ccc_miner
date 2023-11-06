"""
Test script to properly fit Q-gaussian to WS data from the SPS 
"""
import matplotlib.pyplot as plt
from pathlib import Path
from ccc_miner import WS

# Test data location 
data_folder = Path(__file__).resolve().parent.joinpath('../data').absolute()

# Test data corresponds to this logbook entry:
# https://logbook.cern.ch/elogbook-server/GET/showEventInLogbook/3835189
parquet_file = '{}/test_data/WS_2023.09.22.16.08.02.376100.parquet'.format(data_folder)

# Instantiate class 
ws = WS(parquet_file)

# First test 
figure_X, n_emittances_X, sigmas_raw_X, acqTime, ctime_s, Q_values_x = ws.fit_Gaussian_To_and_Plot_Relevant_Profiles(also_fit_Q_Gaussian=True)
figure_Y, n_emittances_Y, sigmas_raw_Y, acqTime, ctime_s, Q_values_y = ws.fit_Gaussian_To_and_Plot_Relevant_Profiles(plane='Y', also_fit_Q_Gaussian=True) 
plt.show()

