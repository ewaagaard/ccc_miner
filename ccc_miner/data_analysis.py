"""
Module for data analysis classes to conveniently handle data from SPS (FBCT and WS )
"""
import json
from ccc_miner import WS, FBCT, plot_settings
import os 
import numpy as np
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Calculate the absolute path to the data folder relative to the module's location
data_folder = Path(__file__).resolve().parent.joinpath('../data').absolute()

class Analyze_WireScanners():
    """Class to process folder of wire scanner data files"""
    def __init__(self, ws_data_path, no_profile_per_scan=0):
        self.folder = ws_data_path
        self.no_profile_per_scan = no_profile_per_scan
        
    def plot_and_save_all_WS_emittances(self, 
                                        output_dest='', 
                                        also_fit_Q_Gaussian=False):
        """
        Iterate through a given WS data folder and save beam profile plots
        Saves all emittances and time stamps to a json dict, which it also saves
        also_fit_Q_Gaussian - optional flag (default False)
        Returns: dictionary of emittances 
        """
        # If string is not empty, check if directory exists 
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

        # Add extra key for Q_gaussian values if needed
        if also_fit_Q_Gaussian:
            full_data['q_values_X'] = []
            full_data['q_values_X_mean'] = []
            full_data['q_values_Y'] = []
            full_data['q_values_Y_mean'] = []

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
                            if also_fit_Q_Gaussian:
                                figure_X, n_emittances_X, sigmas_raw_X, timestamp_X, ctime_X, Q_values_X = ws.fit_Gaussian_To_and_Plot_Relevant_Profiles(plane='X', 
                                                                                                                                no_profiles=self.no_profile_per_scan,  
                                                                                                                                figname=f+'X', also_fit_Q_Gaussian=True)
                            else:
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
                                if also_fit_Q_Gaussian:
                                    full_data['q_values_X'].append(Q_values_X.tolist())
                                    full_data['q_values_X_mean'].append(np.mean(Q_values_X))
                            
                        except TypeError:
                            print('Did not find WS data for X in {}'.format(f))
            
                        try:
                            if also_fit_Q_Gaussian:
                                figure_Y, n_emittances_Y, sigmas_raw_Y, timestamp_Y, ctime_Y, Q_values_Y = ws.fit_Gaussian_To_and_Plot_Relevant_Profiles(plane='Y', 
                                                                                                                               no_profiles=self.no_profile_per_scan,
                                                                                                                               figname=f+'Y', also_fit_Q_Gaussian=True) 
                            else:
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
                                if also_fit_Q_Gaussian:
                                    full_data['q_values_Y'].append(Q_values_Y.tolist())
                                    full_data['q_values_Y_mean'].append(np.mean(Q_values_Y))
                            
                        except TypeError:
                            print('Did not find WS data for Y in {}'.format(f))
                    except KeyError:
                        print('\n\nCannot open device data!\n\n')
        
        # Dump whole dictionary to json file 
        with open('{}/full_WS_data.json'.format(output_dest), 'w') as fp:
            json.dump(full_data, fp) 

        return full_data


    def plot_subset_of_WS_data(self, parquet_file_list, output_dest='', save_json=True):
        """Provide list with special subset of parquet files to be analyzed
            Folder already specified when instantiating the class
        """
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
        
        # Iterate over files in list
        for f in parquet_file_list: 
            # Instantiate class, try to load parquet file
            try:
                print('\nOpening {}\n'.format(str(self.folder) + '/' + f))
                ws = WS(str(self.folder) + '/' + f)
                
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
        if save_json:
            with open('{}/full_WS_data.json'.format(output_dest), 'w') as fp:
                json.dump(full_data, fp) 

        return full_data


    def return_full_dict(self, input_dest=''):
        """
        Loads json file and returns dictionary, with timestamps
        Returns: dict
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


    def plot_emittance_over_bunch_number(self, full_data):
        """For a processed WS subset of parquet files
           Params: full_data dict from self.plot_and_save_all_WS_emittances or self.plot_subset_of_WS_data
           Returns plot of emittance as a function of bunch
        """
        # Iterate over vertical emittance:
        no_samples = len(full_data['N_emittances_Y'])
            
        fig, ax = plt.subplots(2, no_samples, figsize=(10, 6), sharex=True, sharey=True)
        for i, eY_set in enumerate(full_data['N_emittances_Y']):
            eY_set = np.flip(eY_set)
            eX_set = np.flip(full_data['N_emittances_X'][i])
            bunch_number_X = np.arange(1, len(eX_set)+1)
            bunch_number_Y = np.arange(1, len(eY_set)+1)
            ax[0, i].bar(bunch_number_X, 1e6 * eX_set, color='cornflowerblue', label='X: ' + full_data['UTC_timestamp_X'][i])
            ax[1, i].bar(bunch_number_Y, 1e6 * eY_set, color='orange', label='Y: ' + full_data['UTC_timestamp_Y'][i])
        #for ax in ax[:, 0]:
        #    ax.set_ylabel("$\epsilon_{x,y}$ [$\mu$m rad]")
        #for ax in ax[1, :]:
        #    ax.set_xlabel("Bunch number")
        #ax.legend(fontsize=12, loc=2)
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        
        """HOW CAN WE MAKE THIS USEFUL?"""
        
        return fig

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
            ax.set_ylim(0.7, ylim)
        ax.legend(loc=2)
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        return fig


    def plot_q_values_over_cycleTime(self, input_dest='', xlim=None, ylim=None):
        """
        Loads json dict and plots average q-values
        Returns: figure
        """
        # Load processed WS data
        full_data = self.return_full_dict(input_dest)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.errorbar(full_data['Ctime_X'], np.array(full_data['q_values_X_mean']), yerr= np.std(full_data['q_values_X'], axis=1), fmt="o", color='teal', label="X")
        ax.errorbar(full_data['Ctime_Y'], np.array(full_data['q_values_Y_mean']), yerr=np.std(full_data['q_values_Y'], axis=1), fmt="o", color='magenta', label="Y")
        ax.set_ylabel("$q$-value")
        ax.set_xlabel("Cycle time [s]")
        if xlim is not None:
            ax.set_xlim(0, xlim)
        if ylim is not None:
            ax.set_ylim(0.6, ylim)
        ax.legend(loc=2)
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        return fig
    
    def return_avg_emit_X_Y_over_cycle(self, input_dest=''):
        full_data = self.return_full_dict(input_dest)
        return full_data['Ctime_X'], full_data['Ctime_Y'], 1e6 * np.array(full_data['N_avg_emitX']), 1e6 * np.array(full_data['N_avg_emitY']), 1e6 * np.std(full_data['N_emittances_X'], axis=1), 1e6 * np.std(full_data['N_emittances_Y'], axis=1)
                                
    
class Analyze_FBCT_data:
    """Class to process folder of FBCT data files"""
    def __init__(self, fbct_data_path):
        self.folder = fbct_data_path
        
    def plot_and_save_all_bunches_FBCT(self, output_dest='', bunch_selector=None):
        """
        Iterate through a given FBCT data folder and saves data of first X bunches over cycle
        Saves bunch intensity and cycle time to a json dict
        
        Parameters: output destination (string) and bunch selector (array of which bunches to save)
        
        Returns: dictionary of emittances 
        """
        # If string is not empty, check if directory exists 
        #if not output_dest:
        os.makedirs(output_dest, exist_ok=True)
        os.makedirs(output_dest + '/FBCT_plots', exist_ok=True)
        
        # Empty dictionary for full data
        full_data = {
                     'UTC_timestamp':[], 
                     'Bunch_intensities':[],
                     'Avg_bunch_intensity' : [],
                     'Spread_bunch_intensity': [],
                     'Ctime': []
                     }
        
        # Walk through data directory 
        for dirpath, dnames, fnames in os.walk(self.folder):
            for f in fnames:
                if f.endswith(".parquet"):
                    
                    # Instantiate class, try to load parquet file
                    try:
                        print('\nOpening {}'.format(dirpath + '/' + f))
                        fbct = FBCT(dirpath + '/' + f)
                        fig = fbct.plot_selected_bunches(bunch_selector, figname = f +' BCT')
                        fig.savefig('{}/FBCT_plots/{}.png'.format(output_dest, os.path.splitext(f)[0]), dpi=250)
                        
                        # Append the data 
                        intensities = fbct.get_intensity_per_bunch(bunch_selector)
                        full_data['UTC_timestamp'].append(fbct.acqTime)
                        full_data['Bunch_intensities'].append(intensities.tolist())
                        full_data['Avg_bunch_intensity'].append(np.mean(intensities, axis=1).tolist())  # average of selected bunches along cycle 
                        full_data['Ctime'].append((fbct.measStamp * 1e-3).tolist()) # cycle time in seconds
                        full_data['Spread_bunch_intensity'].append(np.std(intensities, axis=1).tolist())
                        print('Successfully processed!\n')
                        
                    except (TypeError, KeyError, AttributeError) as e:
                        print('Did not find FBCT data in {}'.format(f))
        
        # Dump whole dictionary to json file 
        with open('{}/full_FBCT_data.json'.format(output_dest), 'w') as fp:
            json.dump(full_data, fp) 

        return full_data
                    
    
    def iterate_parquet_files_and_plot_bunches(self, output_dest='Output'):
    

        # If string is not empty, check if directory exists 
        #if not output_dest:
        os.makedirs(output_dest, exist_ok=True)
        os.makedirs(output_dest + '/FBCT_plots_bunches_over_buckets', exist_ok=True)
    
        # Walk through data directory 
        for dirpath, dnames, fnames in os.walk(self.folder):
            for f in fnames:
                if f.endswith(".parquet"):
    
                    # Instantiate class, try to load parquet file
                    try:
                        print('\nOpening {}'.format(dirpath + '/' + f))
                        fbct = FBCT(dirpath + '/' + f)
                        fig = fbct.plot()
                        fig.savefig('{}/FBCT_plots_bunches_over_buckets/{}.png'.format(output_dest, os.path.splitext(f)[0]), dpi=250)
                        plt.close()
                        
                    except (TypeError, KeyError, AttributeError) as e:
                        print('Did not find FBCT data in {}'.format(f))                    
    
    
    def return_full_dict(self, input_dest=''):
        """
        Loads json file and returns dictionary, with timestamps
        Returns: dict with FBCT data
        """
        print('\nTrying to load data from {}/full_FBCT_data.json'.format(input_dest))
        # Try reading dictionary from json file 
        try:
            with open('{}/full_FBCT_data.json'.format(input_dest), 'r') as fp:
                full_data = json.load(fp)
            print('Loaded!\n')
        except FileNotFoundError:
            print('\nFILE NOT FOUND - check input path!\n')
            return
        
        # Convert timestamp strings to datetime 
        full_data['Timestamp_datetime'] = pd.to_datetime(full_data['UTC_timestamp'])

        return full_data
    
    def plot_avg_intensity_over_cycleTime(self, input_dest='', run_index = None, xlim=None, ylim=None):
        """
        Loads json dict and plots average bunch intensity for selected bunches
        Returns: figure
        """
        # Load processed FBCT data
        start_ind = 2
        full_data = self.return_full_dict(input_dest)
        avg_Nb_over_cycle = np.mean(np.array(full_data['Bunch_intensities']), axis=(0, 2)) # average of selected bunches over all runs at that point in the cycle
        
        # For plotting, select representative run 
        index = run_index if run_index is not None else 1
        print('\nUTC time stamp of data is {}\n'.format(full_data['UTC_timestamp'][index]))
        Nb = full_data['Bunch_intensities'][index]
        Nb_spread = np.std(full_data['Bunch_intensities'][index], axis=1)
        ctime = np.array(full_data['Ctime'][index])[start_ind:]
        Nb = full_data['Bunch_intensities'][index][start_ind:]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(ctime, Nb, marker='o', ms=5, alpha=0.6)
        #ax.errorbar(ctime, avg_Nb, yerr=spread, fmt="o")
        #ax.plot(ctime, Nb, fmt="o", ms=4, alpha = 0.6, label="Avg Nb: {}".format(fig_str))
        ax.set_ylabel("Bunch intensity")
        ax.set_xlabel("Cycle time [s]")
        if xlim is not None:
            ax.set_xlim(0, xlim)
        if ylim is not None:
            ax.set_ylim(0.7, ylim)
        ax.legend(loc=2)
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        return fig, ctime, Nb
    
    
"""
### TO DO - PROCESS FBCT DATA, FIND CLOSEST RECORDING

  --> example of finding closest point in time of two measurements 
import numpy as np

# Assuming you have two dictionaries: full_data and full_data2
# Each dictionary should have a 'timestamp' entry with DatetimeIndex

# Convert the 'timestamp' entries in both dictionaries to numpy arrays
timestamps1 = np.array(full_data['timestamp'])
timestamps2 = np.array(full_data2['timestamp'])

# Initialize lists to store the corresponding indices in each dictionary
closest_indices1 = []
closest_indices2 = []

for timestamp1 in timestamps1:
    # Calculate the time difference between timestamp1 and all timestamps in timestamps2
    time_diff = np.abs(timestamp1 - timestamps2)
    
    # Find the index of the minimum time difference
    closest_index = np.argmin(time_diff)
    
    # Append the indices to the respective lists
    closest_indices1.append(np.where(timestamps1 == timestamp1)[0][0])
    closest_indices2.append(closest_index)

# Now, closest_indices1 and closest_indices2 contain the indices of the closest matching entries in each dictionary.
""" 