"""
Test script to properly read and fBCT file from SPS
"""
import matplotlib.pyplot as plt
from ccc_miner import WS

# Load data into fBCT
parquet_file = 'test_data/WS_2023.09.22.16.08.02.376100.parquet'

# Instantiate class 
ws = WS(parquet_file)

#print(ws.get_beta_x_and_y_at_WS())

ws.fit_Gaussian_To_Relevant_Profiles()

"""
pos_all, prof_all, index = ws.extract_Meaningful_Bunches_profiles()
print(index)

# Plot the positions and profile
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(1,1,1)
for i, pos in enumerate(pos_all):
    ax.plot(pos, prof_all[i], marker='*', markersize=5, linestyle=None) 
ax.set_xlabel('Position (mm)')
ax.set_ylabel('Amplitude (a.u.)')    
plt.show()
#"""