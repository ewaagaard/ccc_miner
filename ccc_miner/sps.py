"""
Class container to process and plot extracted MD data from SPS
"""
import numpy as np
from scipy.special import gamma as Gamma
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import pyarrow.parquet as pq
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

from ccc_miner import plot_settings

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
        self.dpp = 1e-3
        
        # Select correct wire scanner device - will be read from wire scanner data directly
        #self.WS_device_H = WS_device_H
        #self.WS_device_V = WS_device_V
        # OLD WAY DONE FOR 2023 data
        
        # Plot settings
        self._nrows = 1
        self._ncols = 1
        self._sharex = False
        self._sharey = False
        self.cmap = 'Spectral'
        self.stride = stride
        
        # FBCT filling pattern - injection time in seconds for all 4 * 14 bunches injected
        self.inj_times_all = np.reshape(np.array([0., 0., 0., 0., 3.6, 3.6, 3.6, 3.6, 7.2, 7.2, 7.2, 7.2, 10.8, 10.8, 10.8, 10.8, 
                                       14.4, 14.4, 14.4, 14.4, 18., 18., 18., 18., 21.6, 21.6, 21.6, 21.6, 25.2, 25.2, 25.2, 25.2, 
                                       28.8, 28.8, 28.8, 28.8, 32.4, 32.4, 32.4, 32.4, 36., 36., 36., 36., 39.6, 39.6, 39.6, 39.6, 
                                       43.2, 43.2, 43.2, 43.2, 46.8, 46.8, 46.8, 46.8]), (14, 4))
        self.inj_times = self.inj_times_all[:, 0]


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

    ########### FIT FUNCTION - GAUSSIAN ###########
    def Gaussian(self, x, A, mean, sigma, offset):
        """Gaussian fit of the data """
        return A * np.exp(-(x - mean)**2 / (2 * sigma**2)) + offset
    
    
    def fit_Gaussian(self, x_data, y_data, p0 = None):
        """ 
        Fit Gaussian to given X and Y data (numpy arrays)
        Custom guess p0 can be provided, otherwise generate guess
        
        Returns: parameters popt
        """
        
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
    
    ########### FIT FUNCTION - Q-GAUSSIAN ###########
    def _Cq(self, q, margin=5e-4):
        """
        Normalizing constant from Eq. (2.2) in https://link.springer.com/article/10.1007/s00032-008-0087-y
        with a small margin around 1.0 for numerical stability
        """
        if q < (1 - margin):
            Cq = (2 * np.sqrt(np.pi) * Gamma(1.0/(1.0-q))) / ((3.0 - q) * np.sqrt(1.0 - q) * Gamma( (3.0-q)/(2*(1.0 -q))))   
        elif (q > (1.0 - margin) and q < (1.0 + margin)):
            Cq = np.sqrt(np.pi)
        else:
            Cq = (np.sqrt(np.pi) * Gamma((3.0-q)/(2*(q-1.0)))) / (np.sqrt(q-1.0) * Gamma(1.0/(q-1.0)))
        if q > 3.0:
            raise ValueError("q must be smaller than 3!")
        else:
            return Cq
    
    def _eq(self, x, q):
        """ Q-exponential function
            Available at https://link.springer.com/article/10.1007/s00032-008-0087-y
        """
        eq = np.zeros(len(x))
        for i, xx in enumerate(x):
            if ((q != 1) and (1 + (1 - q) * xx) > 0):
                eq[i] = (1 + (1 - q) * xx)**(1 / (1 - q))
            elif q==1:
                eq[i] = np.exp(xx)
            else:
                eq[i] = 0
        return eq
    
    
    def Q_Gaussian(self, x, mu, q, beta, A, C):
        """
        Returns Q-Gaussian from Eq. (2.1) in (Umarov, Tsallis, Steinberg, 2008) 
        available at https://link.springer.com/article/10.1007/s00032-008-0087-y
        """
        Gq =  A * np.sqrt(beta) / self._Cq(q) * self._eq(-beta*(x - mu)**2, q) + C
        return Gq
    
    
    def fit_Q_Gaussian(self, x_data, y_data, q0 = 1.4):
        """
        Fits Q-Gaussian to x- and y-data (numpy arrays)
        Parameters: q0 (starting guess)
        
        Returns fitted parameters poptq and fit errors poptqe
        """
    
        # Test Gaussian fit for the first guess
        popt = self.fit_Gaussian(x_data, y_data) # gives A, mu, sigma, offset
        p0 = [popt[1], q0, 1/popt[2]**2/(5-3*q0), 2*popt[0], popt[3]] # mu, q, beta, A, offset
    
        #min_bounds = (-np.inf, -np.inf, 0.0, -np.inf, -np.inf,-np.inf)
        #max_bounds = (np.inf, 3.0, np.inf, np.inf, np.inf, np.inf)
    
        try:
            poptq, pcovq = curve_fit(self.Q_Gaussian, x_data, y_data, p0)
            poptqe = np.sqrt(np.diag(pcovq))
        except (RuntimeError, ValueError):
            poptq = np.nan * np.ones(len(p0))
            
        return poptq


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
            
        def plot(self, show_plot=False):
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
            if show_plot:
                plt.show()
            return figure
        

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
                with open('{}/SPS_WS_optics.json'.format(seq_folder)) as f:
                    optics_dict = json.load(f)
                
                # Select correct optics for the relevant wire scanner
                optics_ws_X = optics_dict[self.WS_device_H[8:13]]
                optics_ws_Y = optics_dict[self.WS_device_V[8:13]]
                
                print('Optics for\n{}\n{}\n'.format(self.WS_device_H, self.WS_device_V))
                print('Betx = {:.3f}\nBety = {:.3f}\ndx = {:.3f}'.format(optics_ws_X['betx'], optics_ws_Y['bety'], optics_ws_X['dx']))
                
                return optics_ws_X['betx'], optics_ws_Y['bety'], optics_ws_X['dx'] 
            
            except FileNotFoundError:
                print('Optics file not found: need to run find_WS_optics module in data folder first!')
        
        def load_X_Y_data(self, parquet_file):
            # Check beta function
            # Read data from both devices 
            data = pq.read_table(parquet_file).to_pydict()
            devices = list(data.keys())
            
            # Find the correct wire scanner device
            self.WS_device_H = devices[0]
            self.WS_device_V = devices[2]
            print('Selected WS devices:\n{}\n{}'.format(self.WS_device_H, self.WS_device_V))
            
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
                                                min_integral_fraction=0.5,
                                                amplitude_threshold=900, #650,#1200,
                                                max_number_of_bunches=14*4,
                                                ):
            """ 
            Get all WS profiles and positions from chosen PM, only focus on the meaningful ones
            
            Params:
            ------
            data (X or Y, from method self.load_X_Y_data() )
            ws_set ('Set1' (IN, default) or 'Set2' (Out))
            min_integral_fraction (fractional threshold of max integral that each profile needs to be classified as real)
            amplitude threshold (minimum peak amplitude a profile needs to be classified as a readout)
            
            Returns:
            -------
            list(relevant_profile_positions), (relevant_profiles), (index)
            """
            # Extract the single-PM chosen bunches
            profile_position_all_bunches, profile_data_all_bunches = self.getSingle_PM_ProfileData(data, ws_set)
            
            # Select profiles whose amplitude reading are above the threshoold - and below max number of bunches
            relevant_profiles, relevant_profile_positions, integral_values, index = [], [], [], []
            count = 0
            for i, profile in enumerate(profile_data_all_bunches):         
                 if np.max(profile) >= amplitude_threshold and count < max_number_of_bunches:
                     
                     # The peak above amplitude threshold, shift profile to zero-level to calculate integral
                     profile_shifted = profile + np.abs(np.min(profile)) if np.min(profile) < 0 else profile
                     integral_values.append(np.trapz(profile_shifted)) 
                     
                     relevant_profiles.append(profile)
                     relevant_profile_positions.append(profile_position_all_bunches[i])
                     index.append(i)
                     count += 1
            if not relevant_profiles:
                print('\n\nNO RELEVANT PROFILES ABOVE NOISE THRESHOLD EXTRACTED!\n\n')
                pass
            else: 
                # Calculate the threshold for X% of the maximum integral
                threshold = max(integral_values) * min_integral_fraction    
                
                # Select profiles whose integral is above the threshold
                filtered_profiles = []
                filtered_profile_positions = []
                filtered_indices = []
                for i, (profile, position, index) in enumerate(zip(relevant_profiles, relevant_profile_positions, index)):
                    if integral_values[i] >= threshold:
                        filtered_profiles.append(profile)
                        filtered_profile_positions.append(position)
                        filtered_indices.append(index)
                
                if not filtered_profiles:
                    print('\n\nNO RELEVANT PROFILES ABOVE THE INTEGRAL THRESHOLD EXTRACTED!\n\n')
                
                # Sort the filtered profiles by their integral values
                sorted_profiles = sorted(zip(filtered_profiles, filtered_profile_positions, filtered_indices), key=lambda x: np.trapz(x[0]), reverse=True)
    
                # Unzip the sorted profiles into separate lists
                relevant_profiles, relevant_profile_positions, index = zip(*sorted_profiles)
                
                return list(relevant_profile_positions), (relevant_profiles), (index)
        
        
        def fit_Gaussian_To_and_Plot_Relevant_Profiles(self, 
                                                       plane = 'X', 
                                                       ws_set='Set1',
                                                       no_profiles=0,
                                                       figname=None,
                                                       also_fit_Q_Gaussian=False
                                                       ): 
            """ Fit Gaussian to WS data
                Parameters: 
                    plane ('X' or 'Y')
                    which scan: 'Set1' (INSCAN) or 'Set2' (OUTSCAN)
                    no_profiles: how many bunch profiles to include (first X nr of bunches) - default all (0)
                    figname: name of figure
                    also_fit_Q_Gaussian: whether also to fit and return a Q-Gaussian to the profiles (default False)
            """
            # Read data
            data = self.data_X if plane=='X' else self.data_Y
            

            
            try:
                pos_all, prof_all, index = self.extract_Meaningful_Bunches_profiles(data, ws_set)
            except (TypeError, ValueError, KeyError) as e:
                print('Could not extract data for this timestamp')
                return

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
            popts = [] #np.zeros([no_bunches, 4])
            n_emittances, sigmas_raw = [], [] # previously "np.zeros(no_bunches), np.zeros(no_bunches)", but want to exclude nan values
            
            # Also initiate array for Q-Gaussian
            if also_fit_Q_Gaussian:
                popts_Q = [] # np.zeros([no_bunches, 5])
            
            print('Scanning {} FIRST BUNCHES\n'.format(no_bunches))
            for i in range(no_bunches):
           
                pos = pos_all[i]
                profile_data = prof_all[i]
                
                if i % 20 == 0:
                    print('Scanning bunch {}'.format(i+1))
                
                # Check such that the profile data is not mismatching the position data
                if self.pmtSelection == 5:  # "PM_ALL" --> all photomultipliers selected 
                    # Find where peak is, select this value
                    profile_data_allPM = profile_data.copy()
                    pm_set_index = int(np.floor(np.argmax(profile_data_allPM) / len(pos)))
                    interval = np.arange(len(pos)) + len(pos) * pm_set_index
                    print('ALL PHOTOMULTIPLIERS SELECTED - BEST PM with max: {}'.format(pm_set_index+1))
                    profile_data = profile_data[interval]

                # Fit Gaussian and Q-Gaussian if desired 
                popt = self.fit_Gaussian(pos, profile_data)
                
                # Calculate the emittance from beam paramters 
                beta_func = betx if plane == 'X' else bety
                ctime_s = self.acqTimeinCycleX_inScan/1e3 if plane == 'X' else self.acqTimeinCycleY_inScan/1e3
                sigma_raw = np.abs(popt[2]) / 1e3 # in m
                sigma_betatronic = np.sqrt((sigma_raw)**2 - (self.dpp * dx)**2)
                emittance = sigma_betatronic**2 / beta_func 
                nemittance = emittance * self.beta(self.gamma) * self.gamma 
                
                # Check if fit succeeded
                fit_failed = np.isnan(sigma_betatronic)
                
                if not fit_failed:
                    popts.append(popt)
                
                if also_fit_Q_Gaussian:
                    print('Trying to fit Q-Gaussian...')
                    popt_Q = self.fit_Q_Gaussian(pos, profile_data)
                    if not np.isnan(popt_Q[1]):
                        popts_Q.append(popt_Q)                
                
                if not fit_failed:
                    sigmas_raw.append(sigma_raw)
                    n_emittances.append(nemittance)
                print('Bunch {}: Sigma = {:.3f} mm, n_emittance = {:.4f} um rad\n'.format(i+1, 1e3 * sigma_betatronic, 1e6 * nemittance))
                
                # Plot the data and the fitted curve
                if not fit_failed:
                    ax.plot(pos, profile_data, 'b-', label='Data index {}'.format(index[i]))
                    ax.set_xlim(-30, 30)
                    ax.plot(pos, self.Gaussian(pos, *popt), 'r-', label='Fit index {}'.format(index[i]))
                    if also_fit_Q_Gaussian:
                        ax.plot(pos, self.Q_Gaussian(pos, *popt_Q), color='lime', ls='--', label='Q-Gaussian Fit index {}'.format(index[i]))
            
            en_bar = np.mean(n_emittances)
            spread = np.std(n_emittances)
            ax.text(0.89, 0.89, plane, fontsize=35, fontweight='bold', transform=ax.transAxes)
            ax.text(0.02, 0.12, '{}: {} profiles'.format(ws_set, no_bunches), fontsize=13, transform=ax.transAxes)
            ax.text(0.02, 0.92, 'UTC timestamp:\n {}'.format(self.acqTime[plane]), fontsize=10, transform=ax.transAxes)
            ax.text(0.02, 0.8, 'Plane {} average: \n$\epsilon^n$ = {:.3f} +/- {:.3f} $\mu$m rad'.format(plane, 1e6 * en_bar, 1e6 * spread), fontsize=14, transform=ax.transAxes)
            ax.text(0.78, 0.14, 'InScan {}:\nctime = {:.2f} s'.format(plane, ctime_s),
                                                                            fontsize=11,transform=ax.transAxes)
            if also_fit_Q_Gaussian:
                Q_values = np.array(popts_Q)[:, 1]
                #Q_values = Q_values[~np.isnan(Q_values)]  # remove nan values
                ax.text(0.02, 0.49, 'q-value average: \n{:.3f} +/- {:.3f}'.format(np.mean(Q_values), np.std(Q_values)), fontsize=13, transform=ax.transAxes)
            ax.set_xlabel('Position (mm)')
            ax.set_ylabel('Amplitude (a.u.)')    
            figure.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            
            if also_fit_Q_Gaussian:
                return figure, n_emittances, sigmas_raw, self.acqTime[plane], ctime_s, Q_values
            else:
                return figure, n_emittances, sigmas_raw, self.acqTime[plane], ctime_s
     
            
        def plot_emittances_over_injection_time(self):
            """
            Based on injection time and filling pattern of 4 times 14 bunches, 
            generate plot with emittance over injection time from wire scanner data
            """
             
            # Generate figurues of WS and FBCT
            _, n_emittances_X, sigmas_raw_X, timestamp_X, ctime_X = self.fit_Gaussian_To_and_Plot_Relevant_Profiles(plane='X') 
            _, n_emittances_Y, sigmas_raw_Y, timestamp_Y, ctime_Y = self.fit_Gaussian_To_and_Plot_Relevant_Profiles(plane='Y') 
            #fbct.plot()
            
            # If not 56 (4*14) bunches, fill in remaining with nans
            if len(n_emittances_X) != 56:
                fill_array = np.full(56, np.nan)
                fill_array[0:len(n_emittances_X)] = n_emittances_X
                n_emittances_X = fill_array
            if len(n_emittances_Y) != 56:
                fill_array = np.full(56, np.nan)
                fill_array[0:len(n_emittances_Y)] = n_emittances_Y
                n_emittances_Y = fill_array

            # Check average emittance for each batch - 14 injections in total     
            ex_batch = np.reshape(n_emittances_X, (14, 4))
            ey_batch = np.reshape(n_emittances_Y, (14, 4))
            ex_mean = 1e6 * np.mean(ex_batch, axis=1)  # in um rad
            ex_std = 1e6 * np.std(ex_batch, axis=1) # in um rad      
            ey_mean = 1e6 * np.mean(ey_batch, axis=1)  # in um rad
            ey_std = 1e6 * np.std(ey_batch, axis=1) # in um rad     
            
            # Generate figure with emittance X and Y over cycle for first four bunches from 16/10/2023
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.errorbar(self.inj_times, ex_mean, yerr=ex_std,  fmt="*", ms=19, color='k', ecolor='aqua', markerfacecolor='aqua', label='$\epsilon_{x}$ - each batch at 48 s')
            ax.errorbar(self.inj_times, ey_mean, yerr=ey_std,  fmt="*", ms=19, color='k', ecolor='crimson',  markerfacecolor='crimson', label="$\epsilon_{y}$ - each batch at 48 s")
            ax.set_ylabel("$\epsilon_{x,y}$ [$\mu$m rad]")
            ax.set_xlabel("Time after injection [s]")
            #ax.set_xlim(-0.4, 48.2)
            #ax.set_ylim(0.7, 2.5)
            ax.legend(loc=2)
            fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
            plt.show()