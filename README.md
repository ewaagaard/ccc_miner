# CCC Miner

*UNDER CONSTRUCTION*

Software modules to extract and process data from SPS, PS and LEIR from the CERN Control Center (CCC). The `ccc_miner` contains a class for each accelerator, for now only SPS (under construction). The module for each accelerator contains classes for each device used to measure beam parameters for the respective machine, including wire scanners (WS), Beam Current Transformers (BCT), tomoscopes, etc. 

### Quick set-up

When using Python for scientific computing, it is important to be aware of dependencies and compatibility of different packages. This guide gives a good explanation: [Python dependency manager guide](https://aaltoscicomp.github.io/python-for-scicomp/dependencies/#dependency-management). An isolated environment allows installing packages without affecting the rest of your operating system or any other projects. A useful resource to handle virtual environments is [Anaconda](https://www.anaconda.com/) (or its lighter version Miniconda), when once installed has many useful commands of which many can be found in the [Conda cheat sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) 

To directly start calculating different ion performances with the `ccc_miner`, create an isolated virtual environment and perform a local install to use the `ccc_miner` freely. Once the repository is cloned and from inside the `ccc_miner` repository, run in the terminal:

```
conda create --name test_venv python=3.11 numpy pandas scipy matplotlib
conda activate test_venv
python -m pip install -e ccc_miner
```
The virtual environment can also be installed directly from the `requirements.txt`: `python -m pip install -r requirements.txt`

Or within conda: `conda env create -f environment.yml`


## SPS 

The `SPS` is a parent class, located in `ccc_miner/sps`, from which all device classes inherits properties and methods. The class contains device classes to process data:
- `WS` (Wire Scanner) to measure beam sizes and emittance
- `FBCT` (Fast BCT) to measure beam intensity for each bunch along the cycle

### Wire scanners 

The method `ws.fit_Gaussian_To_and_Plot_Relevant_Profiles()` (default plane is `X`) fits a Gaussian to all beam profiles above a given amplitude threshold. An example to process wire scanner data is located in `ccc_miner/tests/test_sps_ws.py`. Given a path to a raw parquet file from the SPS Wire Scanner, the `ccc_miner` provides tools to process the data, fit Gaussian beam profiles and extract beam emittances: 
```
import matplotlib.pyplot as plt
from pathlib import Path
from ccc_miner import WS

# Specify data location (wherever files are located) - use the Path library 
data_folder = Path(__file__).resolve().parent.joinpath('../data').absolute()

# Test data corresponds to this logbook entry:
# https://logbook.cern.ch/elogbook-server/GET/showEventInLogbook/3835189
parquet_file = '{}/test_data/WS_2023.09.22.16.08.02.376100.parquet'.format(data_folder)

# Instantiate class 
ws = WS(parquet_file)

# Retrieve emittances, beam sizes and figures (default is X plane) 
figure_X, n_emittances_X, sigmas_raw_X, acqTime = ws.fit_Gaussian_To_and_Plot_Relevant_Profiles() 
figure_Y, n_emittances_Y, sigmas_raw_Y, acqTime = ws.fit_Gaussian_To_and_Plot_Relevant_Profiles(plane='Y') 
plt.show()
```

![image](https://github.com/ewaagaard/ccc_miner/assets/68541324/dd40a3e7-387c-4777-81fc-80e7508ca11e)

For every parquet file from the wire scanner, there is also the option to fit a [Q-Gaussian](https://en.wikipedia.org/wiki/Q-Gaussian_distribution) to investigate how many particles that are contained in the tails. Setting `also_fit_Q_Gaussian=True` generates Q-Gaussian fits in green and the average Q-values. 
```
figure_X, n_emittances_X, sigmas_raw_X, acqTime, Q_values = ws.fit_Gaussian_To_and_Plot_Relevant_Profiles(also_fit_Q_Gaussian=True)
```
![2023 10 16 10 59 19 964739_X](https://github.com/ewaagaard/ccc_miner/assets/68541324/1e0883e7-d477-45cc-b955-a261dcf27c0b)


### Fast BCT 

The Fast Beam Current Transformer (BCT) records bunch-by-bunch data of all batches injected in the SPS, for each 25 ns slot (bucket), across the cycle. The `FBCT(parquet_file)` takes the raw parquet file as input and can plot the intensity over cycle time simply using the `FBCT.plot()` method. 
```
from pathlib import Path
from ccc_miner import FBCT

# Test data location 
data_folder = Path(__file__).resolve().parent.joinpath('../data').absolute()

# Load data into fBCT
parquet_file = '{}/test_data/FBCT_2023.09.22.16.07.26.612797.parquet'.format(data_folder)

# Instantiate class 
fbct = FBCT(parquet_file)
fbct.plot()
```

![image](https://github.com/ewaagaard/ccc_miner/assets/68541324/ac2cb799-9a34-48c0-aad0-5ea5a9238851)


### Data analysis class for wire scanners and FBCT 

Data anlysis classes contained in the module `ccc_miner.data_analysis` provide automatic analysis processes for a given directory of parquet files from the Wire Scanner or FBCT. Dictionaries `full_data` are returned, containing data on emittances, cycle times, acquisition time stamp, intensitities, and so on. They are also saved as json files in a specified output location.  

```
import numpy as np
from ccc_miner import Analyze_WireScanners, Analyze_FBCT_data

# Provide selector array of which wire scanner bunches to study 
no_bunches = 4
selector = np.arange(no_bunches)
output_path = 'Output'

WS = Analyze_WireScanners(folderpath_WS, no_bunches)
full_data_WS = WS.plot_and_save_all_WS_emittances(output, also_fit_Q_Gaussian=True)
fig = WS.plot_emittance_over_cycleTime(output_path)

# Analyze FBCT data, then make plots of bunch intensity over cycle time from the produced json file,
# Among all files in the dictionary, select representative index if desired (if not specified take average of all)
FBCT = Analyze_FBCT_data(folderpath_FBCT)
full_data_FBCT = FBCT.plot_and_save_all_bunches_FBCT(output_path, selector)
fig2, ctime, Nb = FBCT.plot_avg_intensity_over_cycleTime(output_path, run_index = 4)
```
![WS_emittances_from_class_SPS_long_flat_bottom_16_10_2023](https://github.com/ewaagaard/ccc_miner/assets/68541324/50875882-bb36-438b-90ef-a84c850f0d80)

![Avg_Bunch_intensity_SPS_long_flat_bottom_16_10_2023](https://github.com/ewaagaard/ccc_miner/assets/68541324/6eb65a56-3aa1-4a3e-9712-fe0c7fb837c4)


