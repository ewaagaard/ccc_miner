"""
Module for data analysis classes to conveniently handle data from SPS (FBCT and WS )
"""
import json
from ccc_miner import WS
import os 
import numpy as np
from pathlib import Path

# Calculate the absolute path to the data folder relative to the module's location
data_folder = Path(__file__).resolve().parent.joinpath('../data').absolute()

class Analyze_WireScanners():
    """Class to process folder of wire scanner data files"""
    def __init__(self, ws_data_path):
        self.folder = ws_data_path
        
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
                     'UTC_timestamp_Y': [],
                     'N_emittances_Y':[],
                     'N_avg_emitY' : []
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
                            figure_X, n_emittances_X, sigmas_raw_X, timestamp_X = ws.fit_Gaussian_To_and_Plot_Relevant_Profiles(plane='X', figname=f+'X') 
                            figure_X.savefig('{}/Plots/{}_X.png'.format(output_dest, os.path.splitext(f)[0]), dpi=250)
                            
                            # Append data if not all NaN
                            if not np.isnan(n_emittances_X).all():
                                full_data['UTC_timestamp_X'].append(timestamp_X)
                                full_data['N_emittances_X'].append(n_emittances_X.tolist())
                                full_data['N_avg_emitX'].append(np.mean(n_emittances_X))
                            
                        except TypeError:
                            print('Did not find WS data for X in {}'.format(f))
            
                        try:           
                            figure_Y, n_emittances_Y, sigmas_raw_Y,timestamp_Y = ws.fit_Gaussian_To_and_Plot_Relevant_Profiles(plane='Y', figname=f+'Y') 
                            figure_Y.savefig('{}/Plots/{}_Y.png'.format(output_dest, os.path.splitext(f)[0]), dpi=250)
                            
                            # Append data if not all NaN
                            if not np.isnan(n_emittances_Y).all():
                                full_data['UTC_timestamp_Y'].append(timestamp_Y)
                                full_data['N_emittances_Y'].append(n_emittances_Y.tolist())
                                full_data['N_avg_emitY'].append(np.mean(n_emittances_Y))
                            
                        except TypeError:
                            print('Did not find WS data for Y in {}'.format(f))
                    except KeyError:
                        print('\n\nCannot open device data!\n\n')
        
        # Dump whole dictionary to json file 
        with open('{}/full_WS_data.json'.format(output_dest), 'w') as fp:
            json.dump(full_data, fp) 

        return full_data

