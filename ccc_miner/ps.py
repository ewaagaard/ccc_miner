"""
Class container to process and plot MD data from Proton Synchrotron (PS)
"""
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import pyarrow.parquet as pq
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
import pandas as pd

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
        t_end 
        t_injection, 
        number of traces, 
        N_samples
    """ 
    def __init__(self,
                 t_end,
                 t_inj=234,
                 no_traces=400,
                 N_samples=2000
                 ):
        super().__init__()  # instantiate PS class
        
        # Store tomoscope parameters
        self.t_end = t_end
        self.t_inj = t_inj
        self.time = np.linspace(t_inj, t_end, self.no_traces) 
        self.no_traces = no_traces
        self.N_samples = N_samples
        
        
    def convert_dat_to_dataframe(self, 
                                 dat_file,
                                 skip_header=98
                                 ):
        """Returns pandas dataframe from raw dat file"""
        data_raw = np.genfromtxt(dat_file, skip_header=skip_header)
        data = np.reshape(data_raw, (self.no_traces, self.N_samples)) 
        
        
        
        
    def fit_gaussian_to_first_peak(self, df, row_index):
        """Extract bunch length sigma_z from Gaussian of first peak among bunches"""
                
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
        
        fig.savefig('Plots/BL_in_time/{}_curve.png'.format(row_index), dpi=250)
        
        plt.close()

        if managed_fit:
            return sigma, d_sigma