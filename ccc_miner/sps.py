"""
Process and plot extracted MD data from SPS, PS and LEIR, tailoried for each individual format
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
        Default min threshold to consider a slot a successful bunch 
        
        Parameters
            data : raw parquet file 
        """
        def __init__(self, parquet_file, min_intensity = 150):
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
            
            # Extract filling index for bunches
            slip_stacking_ind = self.measStamp < 50e3 # point after which slip stacking happens
            
            # DEFAULT FILLING SCHEME - 7 injections
            # Not always 7 injections, but always 4 bunches per batch
            # In every batch of four bunches, three empty slots. Between every batch, four empty slots 
            # Check the min intensity criteria before slip stacking 
            self.bunch_index = np.where(np.any(self.bunchIntensity[slip_stacking_ind, :] > min_intensity, axis=0))[0]
            
            
        def load_data(self, parquet_file):
            """Load FBCT parquet data file"""
            data = pq.read_table(parquet_file).to_pydict()
            self.d =  data[self.fBCT_device][0]['value']
            
        def plot(self):
            """Generate figure showing intensity per bunch over cycle"""
            figure, axs = self.createSubplots('fBCT', 2)  

            # Generate color map of bunch intensity over 25-ns bucket and cycle time
            im = axs[0].pcolormesh(range(self.n_slots), 1e-3*self.measStamp, 
                    self.bunchIntensity * self.unit, cmap=self.cmap, shading='nearest')  # , vmax=0.2e10 to change intensity scale
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
            figure.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.show()
      

        def get_intensity_per_bunch(self, selector=None):
            """
            Get intensity of selected bunches. 
            Selector is array with bunches, e.g. [0, 1, 2, 3] to get first four Default: all
            Returns: matrix with intensity evolution for each selected bunch 
            """
            # Select correct bunches
            try:
                self.bunch_subset = self.bunch_index if selector is None else self.bunch_index[selector]
                return self.bunchIntensity[:, self.bunch_subset] * self.unit  # in correct unit
            except IndexError:
                print('\nNo sufficiently strong bunch intensities found!\n')
                return
        
        def plot_selected_bunches(self, selector=None, plot_legend=True, figname='fbct'):
            """Plot intensity evolution of specific bunches"""
            selected_bunchIntensities = self.get_intensity_per_bunch(selector)
            ctime = 1e-3*self.measStamp # in seconds 
            
            # Plot these selected bunches - iterate over bunches
            figure, ax = self.createSubplots(figname)
            #ax = axs[0]
            count = 0
            for bunch in selected_bunchIntensities.T:
                ax.plot(ctime, bunch, label='Bunch {}'.format(count + 1))
                count += 1
            ax.set_ylabel('Bunch intensity')
            ax.set_xlabel('Cycle time [s]')
            if plot_legend:
                ax.legend()
            figure.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            return figure        
        

class WS(SPS):
        """
        Wire Scanner - measures beam size at given point in cycle, from which gamma
        can be calculated 
        
        Parameters
            data : raw parquet file 
        """
        def __init__(self, 
                     parquet_file,
                     ):  
            super().__init__()  # instantiate SPS class
            
            # Load data
            self.load_X_Y_data(parquet_file)
            
        def beta(self, gamma):
            """Convert relativistic gamma factor to beta factor"""
            beta = np.sqrt(1.0 - 1/gamma**2)
            return beta
            
        def optics_at_WS(self):
            """Reads betx, bety and dx from json file if exists """
            try:
                f = open('{}/SPS_WS_optics.json'.format(seq_folder))
                optics_dict = json.load(f)
                return optics_dict['betx'], optics_dict['bety'], optics_dict['dx'] 
            except FileNotFoundError:
                print('Optics file not found: need to run find_WS_optics module in data folder first!')
        
        def load_X_Y_data(self, parquet_file):
            # Check beta function
            # Read data from both devices 
            data = pq.read_table(parquet_file).to_pydict()
            self.data_X = data[self.WS_device_H][0]['value']
            self.data_Y = data[self.WS_device_V][0]['value']
            
            # Check if different acquisition times in X and Y, otherwise pick X
            try:
                self.acqTime =  {'X': self.data_X['acqTime'], 'Y': self.data_Y['acqTime']}
                exists_X_data, exists_Y_data = True, True 
            except TypeError:
                try:
                    self.acqTime =  {'X': self.data_X['acqTime']}
                    exists_X_data, exists_Y_data = True, False 
                except TypeError:
                    try:
                        self.acqTime =  {'Y': self.data_Y['acqTime']}
                        exists_X_data, exists_Y_data = False, True
                    except TypeError:
                        exists_X_data, exists_Y_data = False, False
                        print('\nDATA DOES NOT EXIST IN NEITHER PLANE!\n')
                        return
                    
            self.gamma_cycle = data['SPSBEAM/GAMMA'][0]['value']['JAPC_FUNCTION']  # IS THIS FOR PROTONS OR IONS?

            # Read processing parameters 
            data = self.data_X if exists_X_data else self.data_Y
            
            self.pmtSelection = data['pmtSelection']['JAPC_ENUM']['code'] # Selected photo-multiplier (PM)
            self.nbAcqChannels = data['nbAcqChannels'] # number of Acq channels (usually 1)
            self.delays = data['delays'][0] / 1e3 
            self.nBunches = len(data['bunchSelection'])     
            self.acqTimeinCycleX_inScan = self.data_X['acqTimeInCycleSet1'] if exists_X_data else np.nan
            self.acqTimeinCycleY_inScan = self.data_Y['acqTimeInCycleSet1'] if exists_Y_data else np.nan
            
            # Find relativistic gamma - remove all gamma cycle data points where Y is possibly zero
            gamma_raw_Y = np.array(self.gamma_cycle['Y'], dtype=np.float64)
            gamma_cycle_Y = gamma_raw_Y[np.isnan(gamma_raw_Y) == 0.]
            gamma_cycle_X = np.array(self.gamma_cycle['X'])[np.isnan(gamma_raw_Y) == 0.]                         
            self.gamma = np.interp(self.delays, gamma_cycle_X, gamma_cycle_Y)  # find gamma at correct delay
            

        def getSingle_PM_ProfileData(self, data, ws_set='Set1', pmtSelection=None):  
            """ Extract Wire Scanner profile data - from chosen single photo-multipliers (PM) 
                
                Default is INSCAN (Set1) but also OUTSCAN (Set2) can be used
                
                Setting bunch selectors e.g. 1-20; 920-924 means that all bunches are included,
                so in this case 41 bunches - but not all bunches are the actual meaningful bunches
             """
            if data is None:
                raise ValueError('Have to provide input data!')
            
            # Extract relevant data
            profile_position_all_bunches = np.array(data['projPosition' + ws_set])
            profile_data_all_bunches = np.array(data['projData' + ws_set]) 
            
            return profile_position_all_bunches, profile_data_all_bunches
            
        
        def extract_Meaningful_Bunches_profiles(self, data, 
                                                ws_set='Set1',
                                                no_bunches=40,
                                                amplitude_threshold=1200,
                                                max_profiles=56
                                                ):
            """ Get all WS profiles and positions from chosen PM, only focus on the meaningful ones"""
            
            # Extract the single-PM chosen bunches
            profile_position_all_bunches, profile_data_all_bunches = self.getSingle_PM_ProfileData(data, ws_set)
            
            # Select profiles whose amplitude reading are above the threshoold
            relevant_profiles, relevant_profile_positions, index = [], [], []
            for i, profile in enumerate(profile_data_all_bunches):         
                 if np.max(profile) >= amplitude_threshold:
                     relevant_profiles.append(profile)
                     relevant_profile_positions.append(profile_position_all_bunches[i])
                     index.append(i)
            if not relevant_profiles:
                print('\n\nNO RELEVANT PROFILES ABOVE NOISE THRESHOLD EXTRACTED!\n\n')
            
            # REDO THIS, BUT COMPARING THE INTEGRALS! NEED FIT FOR THIS 
            """
            # If the number of relevant profiles exceeds max_profiles, select the top max_profiles based on amplitude
            if len(relevant_profiles) > max_profiles:
                # Sort profiles by their maximum amplitude
                sorted_profiles = sorted(zip(relevant_profiles, relevant_profile_positions, index), key=lambda x: np.max(x[0]), reverse=True)
                sorted_profiles = sorted_profiles[:max_profiles]
                
                # Unzip the sorted profiles into separate lists
                relevant_profiles, relevant_profile_positions, index = zip(*sorted_profiles)
            """
            
            return list(relevant_profile_positions), (relevant_profiles), (index)
        
        
        def fit_Gaussian_To_and_Plot_Relevant_Profiles(self, 
                                                       plane = 'X', 
                                                       ws_set='Set1',
                                                       no_profiles=0,
                                                       figname=None
                                                       ): 
            """ Fit Gaussian to WS data
                Parameters: 
                    plane ('X' or 'Y')
                    which scan: 'Set1' (INSCAN) or 'Set2' (OUTSCAN)
                    no_profiles: how many bunch profiles to include (first X nr of bunches) - default all (0)
            """
            # Read data
            data = self.data_X if plane=='X' else self.data_Y
            
            # If all bunches selected, then set much lower amplitude threshhold
            if no_profiles==0:
                pos_all, prof_all, index = self.extract_Meaningful_Bunches_profiles(data, ws_set, amplitude_threshold=-150)
            else:
                pos_all, prof_all, index = self.extract_Meaningful_Bunches_profiles(data, ws_set)

            # Initiate figure
            if figname is None:
                figname = 'BWS {}'.format(plane)
            figure, ax = self.createSubplots(figname)  
            
            # Collect beam parameter info
            betx, bety, dx = self.optics_at_WS()
            
            print('\nUTC TIMESTAMP: {}'.format(self.acqTime[plane]))
            print('\nFitting Gaussians, for beam gamma = {:.4f}\n'.format(self.gamma))
                        
            # Fit Gaussian to relevant profiles - either scan through all or up to given number
            if no_profiles != 0 and prof_all: 
                no_bunches = no_profiles  # custom number of profiles
            elif no_profiles == 0 and prof_all:
                no_bunches =  len(pos_all)  # not selected number of bunches - select all relevant
            else:
                print('No relevant profiles, not fitting Gaussian!\n')    
                return
            
            # Initialize empty arrays
            popts = np.zeros([no_bunches, 4])
            n_emittances, sigmas_raw = np.zeros(no_bunches), np.zeros(no_bunches)
            
            print('Scanning {} FIRST BUNCHES\n'.format(no_bunches))
            for i in range(no_bunches):
           
                pos = pos_all[i]
                profile_data = prof_all[i]
                
                # Check such that the profile data is not mismatching the position data
                if self.pmtSelection == 5:  # "PM_ALL" --> all photomultipliers selected 
                    print('\nALL PHOTOMULTIPLIERS SELECTED - BETTER ONLY WITH ONE!\n')
                    return

                popt = self.fit_Gaussian(pos, profile_data)
                popts[i, :] = popt
                
                # Calculate the emittance from beam paramters 
                beta_func = betx if plane == 'X' else bety
                ctime_s = self.acqTimeinCycleX_inScan/1e3 if plane == 'X' else self.acqTimeinCycleY_inScan/1e3
                sigma_raw = popts[i, 2] / 1e3 # in m
                sigma_betatronic = np.sqrt((sigma_raw)**2 - (self.dpp * dx)**2)
                emittance = sigma_betatronic**2 / beta_func 
                nemittance = emittance * self.beta(self.gamma) * self.gamma 
                sigmas_raw[i] = sigma_raw
                n_emittances[i] = nemittance
                print('Bunch {}: Sigma = {:.3f} mm, n_emittance = {:.4f} um rad\n'.format(i+1, 1e3 * sigma_betatronic, 1e6 * nemittance))
                
                # Plot the data and the fitted curve
                ax.plot(pos, profile_data, 'b-', label='Data index {}'.format(index[i]))
                ax.set_xlim(-30, 30)
                ax.plot(pos, self.Gaussian(pos, *popt), 'r-', label='Fit index {}'.format(index[i]))
            
            en_bar = np.mean(n_emittances)
            spread = np.std(n_emittances)
            ax.text(0.89, 0.89, plane, fontsize=35, fontweight='bold', transform=ax.transAxes)
            ax.text(0.02, 0.12, '{}: {} profiles'.format(ws_set, no_bunches), fontsize=13, transform=ax.transAxes)
            ax.text(0.02, 0.92, 'UTC timestamp:\n {}'.format(self.acqTime[plane]), fontsize=10, transform=ax.transAxes)
            ax.text(0.02, 0.8, 'Plane {} average: \n$\epsilon^n$ = {:.3f} +/- {:.3f} $\mu$m rad'.format(plane, 1e6 * en_bar, 1e6 * spread), fontsize=14, transform=ax.transAxes)
            ax.text(0.78, 0.14, 'InScan {}:\nctime = {:.2f} s'.format(plane, ctime_s),
                                                                            fontsize=11,transform=ax.transAxes)

            ax.set_xlabel('Position (mm)')
            ax.set_ylabel('Amplitude (a.u.)')    
            figure.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            
            return figure, n_emittances, sigmas_raw, self.acqTime[plane], ctime_s
     