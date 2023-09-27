"""
Process and plot extracted MD data from SPS, PS and LEIR, tailoried for each individual format
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pyarrow.parquet as pq

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
        f, axs = plt.subplots(nrows,ncols, num=num,
                    sharex=sharex, sharey=sharey, *args,**kwargs)
        self.axs = axs
        return f, axs


    def FBCT(self, parquet_file):
        """
        The fast BCT registers bunch-by-bunch intensity - can extract every single bunch
        
        Parameters
        ----------
        data : raw parquet file 

        Returns
        -------
        Plots and ...
        """
        
        # Generate figure
        figure, axs = self.createSubplots('fBCT', 2)  
        
        # Load data 
        data = pq.read_table(parquet_file).to_pydict()
        d =  data[self.fBCT_device][0]['value']
        
        # Find which buckets that are filled and measurements over cycle
        fillingPattern = np.array(d['fillingPattern'])
        last_measStamp_with_beam = max(sum(fillingPattern))
        filledSlots = np.where(np.sum(fillingPattern,axis=0))[0]
        unit = np.power(10, d['bunchIntensity_unitExponent'])
        bunchIntensity = np.array(d['bunchIntensity'])
        n_meas_in_cycle, n_slots = bunchIntensity.shape
        
        # Generate color map of bunch intensity 
        measStamp = np.array(d['measStamp'])
        im = axs[0].pcolormesh(range(n_slots), 1e-3*measStamp, 
                bunchIntensity * unit, cmap=self.cmap, shading='nearest')
        cb1 = plt.colorbar(im,ax=axs[0])
        cb1.set_label('Intensity')
        cmap = matplotlib.cm.get_cmap(self.cmap)
        
        indcs = np.arange(0, d['nbOfMeas'], self.stride)
        for i,indx in enumerate(indcs):
            bunch_intensities = bunchIntensity[indx, :] * unit
            c = cmap(float(i)/len(indcs))
            axs[1].plot(bunch_intensities, color=c)
        sm = plt.cm.ScalarMappable(cmap=cmap, 
            norm=plt.Normalize(vmin=min(1e-3*d['measStamp']), vmax=max(1e-3*d['measStamp'])))
        cb2 = plt.colorbar(sm,ax=axs[1])
        cb2.set_label('Cycle time (s)')
    
       # axs[0].set_title(self.generateTitleStr(data[self.device]), fontsize=10)
        if d['beamDetected']:
            axs[1].set_xlim(min(filledSlots)-20,max(filledSlots)+20)
            axs[0].set_ylim(1e-3*d['measStamp'][0], 
                1e-3*d['measStamp'][last_measStamp_with_beam+2])
        self.axs[0].set_ylabel('Cycle time (s)')
        self.axs[1].set_xlabel('25 ns slot')
        self.axs[1].set_ylabel('Intensity')
        plt.tight_layout()
        plt.show()
        #self.drawFigure()
        