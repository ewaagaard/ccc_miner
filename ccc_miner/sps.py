"""
Process and plot extracted MD data from SPS, PS and LEIR, tailoried for each individual format
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import pyarrow.parquet as pq
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

import xtrack as xt

# Calculate the absolute path to the data folder relative to the module's location
test_data_folder = Path(__file__).resolve().parent.joinpath('../tests/test_data').absolute()
test_seq_folder = Path(__file__).resolve().parent.joinpath('../tests/test_sequence').absolute()

class SPS():
    """
    Plotting and data extraction class for different devices in the SPS: fast BCT for intensity, 
    Wire Scanner for emittance
    - default device names for fBCT and WS is provided
    """
    
    def __init__(self,
                 fBCT_device = 'SPS.BCTW.31931/Acquisition',
                 WS_device_H = 'SPS.BWS.41677.H/Acquisition',
                 WS_device_V = 'SPS.BWS.41678.V/Acquisition',
                 stride = 5
                 ):
        self.fBCT_device = fBCT_device
        self.WS_device_H = WS_device_H
        self.WS_device_V = WS_device_V
        self.dpp = 1e-3
        
        # Plot settings
        self._nrows = 1
        self._ncols = 1
        self._sharex = False
        self._sharey = False
        self.cmap = 'Spectral'
        self.stride = stride

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


class FBCT(SPS):
        """
        The fast BCT registers bunch-by-bunch intensity - can extract every single bunch
        
        Parameters
            data : raw parquet file 
        """
        def __init__(self, parquet_file):
            super().__init__()  # instantiate SPS class
            
            # Load data 
            self.load_data(parquet_file)
            
            # Process data - find which buckets that are filled and measurements over cycle
            self.fillingPattern = np.array(self.d['fillingPattern'])
            self.last_measStamp_with_beam = max(sum(self.fillingPattern))
            self.filledSlots = np.where(np.sum(self.fillingPattern, axis=0))[0]  # where bunches are
            self.unit = np.power(10, self.d['bunchIntensity_unitExponent'])
            self.bunchIntensity = np.array(self.d['bunchIntensity'])  # 2D matrix
            self.n_meas_in_cycle, self.n_slots = self.bunchIntensity.shape
            self.measStamp = np.array(self.d['measStamp'])
            self.nbOfMeas = self.d['nbOfMeas']
            self.acqTime =  self.d['acqTime']
            self.beamDetected =self.d['beamDetected']
            
        def load_data(self, parquet_file):
            """Load FBCT parquet data file"""
            data = pq.read_table(parquet_file).to_pydict()
            self.d =  data[self.fBCT_device][0]['value']
            
        def plot(self):
            """Generate figure showing intensity per bunch over cycle"""
            figure, axs = self.createSubplots('fBCT', 2)  

            # Generate color map of bunch intensity over 25-ns bucket and cycle time
            im = axs[0].pcolormesh(range(self.n_slots), 1e-3*self.measStamp, 
                    self.bunchIntensity * self.unit, cmap=self.cmap, shading='nearest')
            cb1 = plt.colorbar(im,ax=axs[0])
            cb1.set_label('Intensity')
            cmap = matplotlib.cm.get_cmap(self.cmap)
            
            # Generate intensity curve over slot, showing in color evolution over cycle time
            indcs = np.arange(0, self.nbOfMeas, self.stride)
            for i,indx in enumerate(indcs):
                bunch_intensities = self.bunchIntensity[indx, :] * self.unit
                c = cmap(float(i)/len(indcs))
                axs[1].plot(bunch_intensities, color=c)
            sm = plt.cm.ScalarMappable(cmap=cmap, 
                norm=plt.Normalize(vmin=min(1e-3 * self.measStamp), vmax=max(1e-3 * self.measStamp)))
            cb2 = plt.colorbar(sm, ax=axs[1])
            cb2.set_label('Cycle time (s)')
        
            axs[0].set_title('{} at {}'.format(self.fBCT_device, self.acqTime), fontsize=10)
            if self.beamDetected:
                axs[1].set_xlim(min(self.filledSlots)-20,max(self.filledSlots)+20)
                axs[0].set_ylim(1e-3 * self.measStamp[0], 
                    1e-3*self.measStamp[self.last_measStamp_with_beam+2])
            self.axs[0].set_ylabel('Cycle time (s)')
            self.axs[1].set_xlabel('25 ns slot')
            self.axs[1].set_ylabel('Intensity')
            plt.tight_layout()
            plt.show()
      

class WS(SPS):
        """
        Wire Scanner - measures beam size at given point in cycle, from which gamma
        can be calculated 
        
        Parameters
            data : raw parquet file 
        """
        def __init__(self, parquet_file, pmtSelection=2):
            super().__init__()  # instantiate SPS class
            
            # Load data
            self.load_X_Y_data(parquet_file)
            
        def beta(self, gamma):
            """Convert relativistic gamma factor to beta factor"""
            beta = np.sqrt(1.0 - 1/gamma**2)
            return beta
            
        def load_X_Y_data(self, parquet_file):
            # Check beta function
            # Read data from both devices 
            data = pq.read_table(parquet_file).to_pydict()
            self.data_X = data[self.WS_device_H][0]['value']
            self.data_Y = data[self.WS_device_V][0]['value']
            self.gamma_cycle = data['SPSBEAM/GAMMA'][0]['value']['JAPC_FUNCTION']  # IS THIS FOR PROTONS OR IONS?

            # Read processing parameters 
            self.pmtSelection = self.data_X['pmtSelection']['JAPC_ENUM']['code'] # Selected photo-multiplier (PM)
            self.nbAcqChannels = self.data_X['nbAcqChannels'] # number of Acq channels (usually 1)
            self.delays = self.data_X['delays'][0] / 1e3 
            self.nBunches = len(self.data_X['bunchSelection'])
            
            # Find relativistic gamma - remove all gamma cycle data points where Y is possibly zero
            gamma_raw_Y = np.array(self.gamma_cycle['Y'], dtype=np.float64)
            gamma_cycle_Y = gamma_raw_Y[np.isnan(gamma_raw_Y) == 0.]
            gamma_cycle_X = np.array(self.gamma_cycle['X'])[np.isnan(gamma_raw_Y) == 0.]                         
            self.gamma = np.interp(self.delays, gamma_cycle_X, gamma_cycle_Y)  # find gamma at correct delay
            
            
            d = self.data_X
            #print(self.data_X['projDataSet1'][1])
            
        def get_beta_x_and_y_at_WS(self):
            """ Find betatronic functions at location of WS in Twiss table, also the horizontal dispersion"""
            
            line_SPS_Pb = xt.Line.from_json('{}/SPS_2021_Pb_ions_matched_with_RF.json'.format(test_seq_folder))
            line_SPS_Pb.build_tracker()
            twiss0_SPS = line_SPS_Pb.twiss().to_pandas()

            # Find wire scanner location
            betx = twiss0_SPS.betx[twiss0_SPS['name'] == 'bwsa.41420'].values[0]
            bety = twiss0_SPS.bety[twiss0_SPS['name'] == 'bwsa.41420'].values[0]
            dx = twiss0_SPS.dx[twiss0_SPS['name'] == 'bwsa.41420'].values[0]

            return betx, bety, dx


        def getSingle_PM_ProfileData(self, data = None, ws_set='Set1', pmtSelection=None):  
            """ Extract Wire Scanner profile data - from chosen single photo-multipliers (PM) 
            
                Setting bunch selectors e.g. 1-20; 920-924 means that all bunches are included,
                so in this case 41 bunches - but not all bunches are the actual meaningful bunches
             """
            if data is None:
                data = self.data_X # default unless provided 
            
            # Extract relevant data
            profile_position_all_bunches = np.array(data['projPosition' + ws_set])
            profile_data_all_bunches = np.array(data['projData' + ws_set]) 
            
            return profile_position_all_bunches, profile_data_all_bunches
            
        
        def extract_Meaningful_Bunches_profiles(self, data = None, no_bunches=4, amplitude_threshhold = 2000):
            """ Get all WS profiles and positions from chosen PM, only focus on the meaningful ones"""
            
            # Extract the single-PM chosen bunches
            profile_position_all_bunches, profile_data_all_bunches = self.getSingle_PM_ProfileData(data)
            
            # Select profiles whose amplitude reading are above the threshoold
            relevant_profiles, relevant_profile_positions, index = [], [], []
            for i, profile in enumerate(profile_data_all_bunches):
                 if np.max(profile) >= amplitude_threshhold:
                     relevant_profiles.append(profile)
                     relevant_profile_positions.append(profile_position_all_bunches[i])
                     index.append(i)
            
            # Alert if there is disagreement
            if no_bunches != len(relevant_profiles):
                print('\nWarning, number of relevant bunches do not seem to match the desired number {}!\n'.format(no_bunches))
            
            return relevant_profile_positions, relevant_profiles, index
        
        def fit_Gaussian_To_Relevant_Profiles(self, data = None, plane = 'X'): 
            """ Fit Gaussian to WS data"""
            
            pos_all, prof_all, index = self.extract_Meaningful_Bunches_profiles(data)
            
            # Initiate figure
            figure, ax = self.createSubplots('BWS')  
            
            # Collect beam parameter info
            popts = np.zeros([len(prof_all), 4])
            n_emittances = np.zeros(len(prof_all))
            betx, bety, dx = self.get_beta_x_and_y_at_WS()
            
            # Fit Gaussian to relevant profiles
            for i, pos in enumerate(pos_all):
                profile_data = prof_all[i]
                popt = self.fit_Gaussian(pos, profile_data)
                popts[i, :] = popt
                
                # Calculate the emittance from beam paramters 
                beta_func = betx if plane == 'X' else bety
                sigma_raw = popts[i, 2] / 1e3 # in m
                sigma_betatronic = np.sqrt((sigma_raw)**2 - (self.dpp * dx)**2)
                emittance = sigma_betatronic**2 / beta_func 
                nemittance = emittance * self.beta(self.gamma) * self.gamma 
                n_emittances[i] = nemittance
                print('\nBunch {}: Sigma = {:.3f} mm, n_emittance = {:.4f} um rad\n'.format(i+1, 1e3 * sigma_betatronic, 1e6 * nemittance))
                
                # Plot the data and the fitted curve
                ax.plot(pos, profile_data, 'b-', label='Data index {}'.format(index[i]))
                ax.plot(pos, self.Gaussian(pos, *popt), 'r-', label='Fit index {}'.format(index[i]))
            
            ax.set_xlabel('Position (mm)')
            ax.set_ylabel('Amplitude (a.u.)')    
            plt.show()
                        
    
            
        # Hannes old version below, not needed for now
        '''
        def get_all_PM_ProfileData(self, ws_set='Set1'):  
            """ Extract Wire Scanner profile data - from all photo-multipliers (PM) 
                if index is set, or 'ALL' (5) or 'AUTO' (0) if those settings were used 
             """
            d = self.data_Xdata = None
            nbAcqChannelsmax = 4            
            nbPts = d['nbPtsPerProjIn%s'%ws_set]
            prof_all_raw = [d['projData%s'%ws_set][i][:] for i in range(self.nBunches)]  # for pandas had to edit index here, with datascout it was [i, :]
            prof_all_tmp = []
            for prof in prof_all_raw:
                prof_all_tmp.append([prof[i*nbPts:(i+1)*nbPts] for i in range(self.nbAcqChannels)])
            prof_all_tmp = np.array(prof_all_tmp)
            pos_all = np.array([d['projPosition%s'%ws_set][i] for i in range(self.nBunches)])#.T
            prof_all = []
           
            # Iterate over all PM profiles, also if several PMs used 
            if self.pmtSelection == 5: # Setting 'PM_ALL', i.e. collecting data from all PMs
                for nAcq in range(nbAcqChannelsmax):
                    prof_all.append(prof_all_tmp[:,nAcq,:])#.T)
            else:
                if self.pmtSelection == 0: # Setting 'PM_AUTO' in WS, i.e. choosing the best photo-multiplier
                    prof_indx = [d['bestChannel%s'%ws_set]-1]
                else:
                    prof_indx = [self.pmtSelection-1]
                for nAcq in range(nbAcqChannelsmax):
                    # Return nan arrays if not chosen PM
                    if nAcq in prof_indx:
                        prof_all.append(prof_all_tmp[:,0,:])
                    else:
                        prof_all.append(np.array([[np.nan]*nbPts]*self.nBunches))
            
            return pos_all, prof_all
            '''