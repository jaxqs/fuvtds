import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import synphot as syn
import matplotlib.gridspec as gridspec
import os
from scipy.stats import binned_statistic

def binned(cenwave, segment, x, y, wgt): 

    # cenwaves --> dictionary
    # 'cenwave': {SEGMENT_A, SEGMENT_B}
    # 'segment': [lower wavelength limit, upper wavelength limit, Angstrom-bin]
    cenwaves = {
        # G160M
        '1533': {'FUVA':[1535.0, 1705.0, 5], 'FUVB': [1345.0, 1515.0, 5]},
        '1577': {'FUVA':[1575.0, 1750.0, 5], 'FUVB': [1385.0, 1560.0, 5]},
        '1611': {'FUVA':[1610.0, 1785.0, 5], 'FUVB': [1420.0, 1590.0, 5]},
        '1623': {'FUVA':[1625.0, 1795.0, 5], 'FUVB': [1435.0, 1605.0, 5]},
        
        # G130M
        '1222': {'FUVA':[1225.0, 1360.0, 5], 'FUVB': [1085.0, 1205.0, 20]},
        '1055': {'FUVA':[1065.0, 1185.0, 20],'FUVB': [910.0, 1030.0, 60]},
        '1096': {'FUVB':[950.0, 1070.0, 20]},
        '1291': {'FUVA':[1290.0, 1430.0, 5], 'FUVB': [1135.0, 1275.0, 5]},
        '1327': {'FUVA':[1325.0, 1470.0, 5], 'FUVB': [1170.0, 1315.0, 5]},

        # G140L 
        '800' : {'FUVA':[920.0, 1800.0, 20]},
        '1105': {'FUVA':[1140.0, 1800.0, 20]},
        '1280': {'FUVA':[1280.0, 1800.0, 20],'FUVB': [1100.0, 1120.0, 20]}
        }

    # only grab data within the established wl ranges in the dictionary
    x_index = np.where((x >= cenwaves[str(cenwave)][segment][0]) &
                          (x <= cenwaves[str(cenwave)][segment][1]))
    
    # summation
    s, edges, _ = binned_statistic(x[x_index], y[x_index], statistic = 'sum', 
                                   bins = (cenwaves[str(cenwave)][segment][1] - 
                                           cenwaves[str(cenwave)][segment][0]) / 
                                           cenwaves[str(cenwave)][segment][2])
    swgt, _ , _ = binned_statistic(x[x_index], wgt[x_index], statistic = 'sum', 
                                   bins = (cenwaves[str(cenwave)][segment][1] - 
                                           cenwaves[str(cenwave)][segment][0]) /
                                           cenwaves[str(cenwave)][segment][2])
    
    # weighted average
    s = s / swgt

    return(s, edges, str(cenwaves[str(cenwave)][segment][2]))

def select_model(target, x):
    if target =='WAVE':
        return
    targs = {
        'GD71': 'gd71_mod_011.fits',
        'WD0308-565': 'wd0308_565_mod_006.fits'
    }
    model_spec = syn.SourceSpectrum.from_file(os.path.join(
        os.environ['PYSYN_CDBS'], 'calspec', targs[target]))
    
    wave = np.arange(x[0]-250, x[-1]+250, 0.1)
    model = model_spec(wave, flux_unit=syn.units.FLAM)

    return (wave, model)

if __name__ == "__main__":
    # change these as needed
    LP = 'LP6'
    PID = '16830'

    # the directory where the data is stored
    data_dir = glob.glob('/grp/hst/cos2/new_TDSTAB_postgeo/DATA/calibrated/'+LP+'/'+PID+'/*_x1d.fits')

    for file in data_dir:
        hdr0 = fits.getheader(file, 0)
        data = fits.getdata(file, 1)

        flux, edges, _ = binned(
            hdr0['cenwave'],
            hdr0['segment'],
            data['wavelength'],
            data['flux'],
            data['DQ_WGT']
            )
        wl = edges[:-1]+np.diff(edges)/2

        wave, model = select_model(hdr0['TARGNAME'], data['WAVELENGTH'][0])

        fig = plt.figure(tight_layout=True)
        gs  = gridspec.GridSpec(1,2)

        #plot 1 - binned scatter
        ax = fig.add_subplot(gs[0,:-1])
        ax.semilogy(wl, flux, ls ='', marker ='o', color='C1', label='new')
        ax.semilogy(wave, model, color='black', label='model')
        

        # plot 2 - histogram
        ax = fig.add_subplot(gs[0:,-1])
        ax.plot(wave, model, color='black')

        plt.show()