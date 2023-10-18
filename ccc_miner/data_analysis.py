"""
Module for data analysis classes to conveniently handle data from SPS (FBCT and WS )
"""
import json
from ccc_miner import WS
import os 
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Calculate the absolute path to the data folder relative to the module's location
data_folder = Path(__file__).resolve().parent.joinpath('../data').absolute()

#### PLOT SETTINGS #######
SMALL_SIZE = 16
MEDIUM_SIZE = 19
BIGGER_SIZE = 21
plt.rcParams["font.family"] = "serif"
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
colors = ['green', 'blue', 'purple', 'brown', 'teal', 'coral', 'cyan', 'darkred']   


class Analyze_WireScanners():
    """Class to process folder of wire scanner data files"""
    def __init__(self, ws_data_path, no_profile_per_scan=0):
        self.folder = ws_data_path
        self.no_profile_per_scan = no_profile_per_scan
        
    def plot_and_save_all_WS_emittances(self, output_dest=''):
        """
        Iterate through a given WS data folder and save beam profile plots
        Saves all emittances and time stamps to a json dict, which it also saves 
        Returns: dictionary of emittances 
        """
        # If string is not empty, check if directory exists 
        #if not output_dest:
        os.makedirs(output_dest, exist_ok=True)
        os.makedirs(output_dest + '/Plots', exist_ok=True)
            
        # Empty dictionary for full data
        full_data = {
                     'UTC_timestamp_X':[], 
                     'N_emittances_X':[],
                     'N_avg_emitX' : [],
                     'Ctime_X': [],
                     'UTC_timestamp_Y': [],
                     'N_emittances_Y':[],
                     'N_avg_emitY' : [],
                     'Ctime_Y': []
                     }

        # Walk through data directory 
        for dirpath, dnames, fnames in os.walk(self.folder):
            for f in fnames:
                if f.endswith(".parquet"):
                    
                    # Instantiate class, try to load parquet file
                    try:
                        
                        print('\nOpening {}\n'.format(dirpath + '/' + f))
                        ws = WS(dirpath + '/' + f)
                        
                        # First test
                        try:
                            figure_X, n_emittances_X, sigmas_raw_X, timestamp_X, ctime_X = ws.fit_Gaussian_To_and_Plot_Relevant_Profiles(plane='X', 
                                                                                                                                no_profiles=self.no_profile_per_scan,  
                                                                                                                                figname=f+'X') 
                            figure_X.savefig('{}/Plots/{}_X.png'.format(output_dest, os.path.splitext(f)[0]), dpi=250)
                            
                            # Append data if not all NaN
                            if not np.isnan(n_emittances_X).all():
                                full_data['UTC_timestamp_X'].append(timestamp_X)
                                full_data['N_emittances_X'].append(n_emittances_X.tolist())
                                full_data['N_avg_emitX'].append(np.mean(n_emittances_X))
                                full_data['Ctime_X'].append(ctime_X)
                            
                        except TypeError:
                            print('Did not find WS data for X in {}'.format(f))
            
                        try:           
                            figure_Y, n_emittances_Y, sigmas_raw_Y, timestamp_Y, ctime_Y = ws.fit_Gaussian_To_and_Plot_Relevant_Profiles(plane='Y', 
                                                                                                                               no_profiles=self.no_profile_per_scan,
                                                                                                                               figname=f+'Y') 
                            figure_Y.savefig('{}/Plots/{}_Y.png'.format(output_dest, os.path.splitext(f)[0]), dpi=250)
                            
                            # Append data if not all NaN
                            if not np.isnan(n_emittances_Y).all():
                                full_data['UTC_timestamp_Y'].append(timestamp_Y)
                                full_data['N_emittances_Y'].append(n_emittances_Y.tolist())
                                full_data['N_avg_emitY'].append(np.mean(n_emittances_Y))
                                full_data['Ctime_Y'].append(ctime_Y)
                            
                        except TypeError:
                            print('Did not find WS data for Y in {}'.format(f))
                    except KeyError:
                        print('\n\nCannot open device data!\n\n')
        
        # Dump whole dictionary to json file 
        with open('{}/full_WS_data.json'.format(output_dest), 'w') as fp:
            json.dump(full_data, fp) 

        return full_data



    def return_full_dict(self, input_dest=''):
        """
        Loads json file and returns dictionary, with timestamps
        Returns: figure
        """
        print('\nTrying to load data from {}/full_WS_data.json\n'.format(input_dest))
        # Try reading dictionary from json file 
        try:
            with open('{}/full_WS_data.json'.format(input_dest), 'r') as fp:
                full_data = json.load(fp)
        except FileNotFoundError:
            print('\nFILE NOT FOUND - check input path!\n')
            return
        
        # Convert timestamp strings to datetime 
        full_data['TimestampX_datetime'] = pd.to_datetime(full_data['UTC_timestamp_X'])
        full_data['TimestampY_datetime'] = pd.to_datetime(full_data['UTC_timestamp_Y'])

        return full_data


    def plot_emittance_over_cycleTime(self, input_dest='', xlim=None, ylim=None):
        """
        Loads json dict and plots average emittances
        Returns: figure
        """
        # Load processed WS data
        full_data = self.return_full_dict(input_dest)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(full_data['Ctime_X'], 1e6 * np.array(full_data['N_avg_emitX']), yerr=1e6 * np.std(full_data['N_emittances_X'], axis=1), fmt="o", label="Hor. Emittance")
        ax.errorbar(full_data['Ctime_Y'], 1e6 * np.array(full_data['N_avg_emitY']), yerr=1e6 * np.std(full_data['N_emittances_Y'], axis=1), fmt="o", label="Ver. Emittance")
        ax.set_ylabel("$\epsilon_{x,y}$ [$\mu$m rad]")
        ax.set_xlabel("Cycle time [s]")
        if xlim is not None:
            ax.set_xlim(0, xlim)
        if ylim is not None:
            ax.set_ylim(0, ylim)
        ax.legend(loc=2)
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        plt.show()