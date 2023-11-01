"""
Class container to process and plot MD data from Proton Synchrotron (PS)
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import pandas as pd
import os

from scipy.constants import c as c_light
from ccc_miner import plot_settings
import seaborn as sns

# Calculate the absolute path to the data folder relative to the module's location
data_folder = Path(__file__).resolve().parent.joinpath('../data').absolute()
seq_folder = Path(__file__).resolve().parent.joinpath('../data/sequence_files').absolute()

class PS():
    """Parent class for PS devices"""
    def __init__(self):
        pass
        
    def createSubplots(self, figname, nrows=1, ncols=1, *args, **kwargs):
        """
        Generate a figure with subplots
        
        Returns
        -------
        fig, axes
        """
        num = figname
        sharex = self._sharex
        sharey = self._sharey
        f, axs = plt.subplots(nrows, ncols, num=num, figsize=(8, 6),
                    sharex=sharex, sharey=sharey, *args,**kwargs)
        self.axs = axs
        return f, axs

    def Gaussian(self, x, A, mean, sigma, offset):
        """Gaussian fit of the data """
        return A * np.exp(-(x - mean)**2 / (2 * sigma**2)) + offset
        
    
    def fit_Gaussian(self, x_data, y_data, p0 = None):
        """ Fit Gaussian from given X and Y data, return parameters"""
        
        # if starting guess not given, provide some qualified guess from data
        if p0 is not None: 
            initial_guess = p0
        else:
            initial_amplitude = np.max(y_data) - np.min(y_data)
            initial_mean = x_data[np.argmax(y_data)]
            initial_sigma = 1.0 # starting guess for now
            initial_offset = np.min(savgol_filter(y_data,21,2))
            
            initial_guess = (initial_amplitude, initial_mean, initial_sigma, initial_offset)
        # Try to fit a Gaussian, otherwise return array of infinity
        try:
            popt, pcov = curve_fit(self.Gaussian, x_data, y_data, p0=initial_guess)
        except (RuntimeError, ValueError):
            popt = np.infty * np.ones(len(initial_guess))
            
        return popt
                 
                 
                 
class PS_Tomoscope(PS):
    """
    PS Tomoscope class
    
    Parameters: (all found from tomoscope application)
        t_span   (has to be specified in ms, can be seen on tomoscope application)
        t_injection, (default 234 ms)
        number of traces, 
        N_samples, 
        beta_at_inj (relativistic beta at injection, value taken for Pb ions)
        bunch_split (True or False)
    """ 
    def __init__(self,
                 t_span,
                 t_inj=234,
                 no_traces=400,
                 N_samples=2000,
                 beta_at_inj = 0.3704,
                 bunch_split=False
                 ):
        super().__init__()  # instantiate PS class
        
        # Store tomoscope parameters
        self.t_end = t_inj + t_span  # t_span seen on tomoscope application
        self.t_inj = t_inj
        self.no_traces = no_traces
        self.time = np.linspace(t_inj, self.t_end, self.no_traces) 
        self.N_samples = N_samples
        self.beta_at_inj = beta_at_inj 
        self.bunch_split = bunch_split
        
    def convert_dat_to_dataframe(self, 
                                 dat_file,
                                 skip_header=98
                                 ):
        """Returns pandas dataframe from raw dat file"""
        data_raw = np.genfromtxt(dat_file, skip_header=skip_header)
        data = np.reshape(data_raw, (self.no_traces, self.N_samples)) 
        df = pd.DataFrame(data)
        df['Cycle_time'] = self.time
        df = df.set_index('Cycle_time')
        return df
        
    def get_heatmap(self, dat_file):
        """Returns seaborn heatmap (waterfall plot) of bunch profiles"""
        df = self.convert_dat_to_dataframe(dat_file)
        
        # Plot the heatmap of the data 
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df, ax=ax, cbar=False)
        ax.set(yticklabels=[])
        fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
        return fig
        
        
    def fit_gaussian_to_first_peak(self, 
                                   df, 
                                   row_index, 
                                   plot_output_dest,
                                   name_string):
        """Extract bunch length sigma_z from Gaussian of first peak among bunches"""
                
        # Check that output directory exists
        os.makedirs(plot_output_dest, exist_ok=True)
        os.makedirs('{}/{}_BL_in_time'.format(name_string, plot_output_dest), exist_ok=True)
        
        # Define data
        data = df.iloc[row_index]
        
        x = np.arange(len(data))
        
        # Initial guess for the amplitude (A) based on the maximum value in the row
        initial_amplitude = np.max(data)
        
        # Initial guess for the mean (center) based on the index of the maximum value
        initial_mean = np.argmax(data) 
        
        # Initial guess for sigma (standard deviation)
        initial_sigma = 50  # You can adjust this value based on your data
        
        # Initial guess for offset (baseline)
        initial_offset = 0.0
        
        initial_guess = (initial_amplitude, initial_mean, initial_sigma, initial_offset)
        
        # Define end index, i.e. remove the data 4 sigmas after the maximum
        start_index = 0
        end_index = initial_mean + int(2 * initial_sigma)
        
        # Fit the Gaussian curve using curve_fit
        try:
            params, covariance = curve_fit(self.Gaussian, x[start_index:end_index], data[start_index:end_index], p0=initial_guess)
            d_sigma = covariance[2,2]

            # Extract the fitted parameters
            amplitude, mean, sigma, offset = params
            
            # Plot the data and the fitted curve
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, data, 'b-', label='Data')
            ax.plot(x[start_index:end_index], self.Gaussian(x[start_index:end_index], *params), 'r-', label='Fit')
            
            # Print the fitted parameters
            print('\nFitting index {}'.format(row_index))
            print(f'Amplitude (A): {amplitude}')
            print(f'Mean: {mean}')
            print('Sigma (standard deviation): {} +/- {}'.format(sigma, covariance))
            print(f'Offset: {offset}')
            plt.legend()
            managed_fit = True
        
        except RuntimeError:
            print('Did not manage to fit!')
        
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, data, 'b-', label='Data')
            managed_fit = False
        
        fig.savefig('{}/{}_BL_in_time/{}_curve.png'.format(name_string, plot_output_dest, row_index), dpi=250)
        
        plt.close()

        if managed_fit:
            return sigma, d_sigma
        
        
    def fit_gaussians_to_all_peaks_and_plot(self, 
                                            dat_file,
                                            name_string,
                                            output_dest='Output_plots', 
                                            sigma_tol=65,
                                            bunch_split_start=90,
                                            bunch_split_stop=104,
                                            energy_ramping=False,
                                            beta_over_cycle=None):
            """
            Iterate through profiles over cycle time and estimate bunch length
            
            Parameters:
                dat_file (raw file from tomoscope),
                name_string (e.g. EARLY or NOMINAL)
                output_dest (output destination for plots)
                sigma_tol (max accepeted value for bunch length in ns)
                bunch_split_start, bunch_split_stop (index for bunch splitting, if happens)
                energy_ramping (flag, if we stay at injection)
                beta_over_cycle  (relativistic beta values over cycle time)
            """
            
            # Make output plot directory
            os.makedirs(output_dest, exist_ok=True)
            print('\nSaving data to {}...\n'.format(output_dest))
            
            # Extract dataframe from file 
            df_sigma_raw = self.convert_dat_to_dataframe(dat_file)
            
            # Initialize empty arrays for row index and sigmas where fit is reasonable
            sigmas = []
            covariances = []
            indices = []
            
            # IF bunch splitting, add index
            if self.bunch_split:    
                bunch_split = []
                bunch_split_index = np.arange(90, 104)  # among the 400 traces, bunch splitting happens heree
            
            # Loop over all profiles of cycle
            i = 0
            for row_index in range(len(df_sigma_raw)):
                sigma, d_sigma = self.fit_gaussian_to_first_peak(df_sigma_raw, 
                                                                 row_index, 
                                                                 output_dest, 
                                                                 name_string)

                # Check if it was possible to fit sigma
                if sigma is not None:
                    if np.abs(sigma) < sigma_tol:
                        sigmas.append(np.abs(sigma))
                        covariances.append(d_sigma)
                        indices.append(row_index)
                        
                        # Add flag for bunch splitting 
                        if self.bunch_split:
                            bunch_split.append(1) if i in bunch_split_index else bunch_split.append(0)
                i += 1
            print(f"Finished! {len(sigmas)} out of {len(df_sigma_raw)} succeeded in fitting!")

            # Create new dataframe, keeping only the points with a good fit
            df = pd.DataFrame()
            df['cycle_time'] = df_sigma_raw.index
            df = df.iloc[indices]
            df['sigma'] = sigmas
            df['covariance'] = covariances
            
            # Add weighted rolling average with 5 points
            WMA_nr = 5
            weights = np.ones(WMA_nr)/WMA_nr  # weights have to add up to 1
            df['sigma_WMA'] = df['sigma'].rolling(WMA_nr).apply(lambda x: np.sum(weights*x))
            
            if self.bunch_split:
                df['bunch_split'] = bunch_split
                
                # Update bunch split index and 
                BS_ind_new = [i for i, e in enumerate(bunch_split) if e != 0]
                bunch_split_start, bunch_split_stop = df['cycle_time'].iloc[BS_ind_new[0]], df['cycle_time'].iloc[BS_ind_new[-1]]
                print("\nBunch split starts at {:.3f} ms and ends at {:.3f} ms".format(bunch_split_start, bunch_split_stop))
                
                # Add index with how many bunches there are 
                df['N_bunches'] = 2 * np.ones(len(df))
                df['N_bunches'].iloc[BS_ind_new[0]:] = 4
                
            # Plot bunch length evolution in ns
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            ax1.axvline(x=self.t_inj, linewidth=2, label='Injection at 235 ms')
            ax1.plot(df['cycle_time'] , df['sigma'], 'b-',  marker='o', label='Measured beam size')
            ax1.plot(df['cycle_time'] , df['sigma_WMA'], ls='--', color='cyan', linewidth = 1.5, label='Sigma rolling average')
            if self.bunch_split:
                ax1.axvspan(bunch_split_start, bunch_split_stop, alpha=0.5, color='red', label='Bunch splitting')
            ax1.set_ylabel(r'$\sigma$ [ns]')
            ax1.set_xlabel('Cycle time [ms]')
            ax1.legend()
            fig1.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            
            # Plot bunch length evolution in m
            beta = self.beta_at_inj if not energy_ramping else beta_over_cycle
            df['sigma [m]'] =  df['sigma'] * 1e-9 * c_light * beta
            df['sigma_WMA [m]'] =  df['sigma_WMA'] * 1e-9 * c_light * beta
            
            # Add dataframe as 
            df.to_csv('{}/{}_bunch_length_data.csv'.format(name_string, output_dest))
            self.df = df
            
        
            fig2, ax2 = plt.subplots(figsize=(9, 6))
            ax2.axvline(x=self.t_inj, linewidth=2, label='Injection at 235 ms')
            ax2.plot(df['cycle_time'] , df['sigma [m]'], 'coral', ls='-',  marker='o', alpha=0.8, label='Measured bunch length')
            ax2.plot(df['cycle_time'] , df['sigma_WMA [m]'], ls='-', color='red', linewidth = 3.5, label='$\sigma$ rolling average')
            if self.bunch_split:
                ax2.axvspan(bunch_split_start, bunch_split_stop, alpha=0.5, color='red', label='Bunch splitting')
            ax2.set_ylabel(r'$\sigma_{z}$ [m]')
            ax2.set_xlabel('Cycle time [ms]')
            ax2.legend()
            fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            
            return fig1, fig2
            
            