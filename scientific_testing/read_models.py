import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import synphot as syn
import matplotlib.gridspec as gridspec
import os

def select_model(target, x):
    if target =='WAVE':
        return
    targs = {
        'GD71': 'gd71_mod_011.fits',
        'WD0308-565': 'wd0308_565_mod_006.fits'
    }
    model_spec = syn.SourceSpectrum.from_file(os.path.join(
        os.environ['PYSYN_CDBS'], 'calspec', targs[target]))
    
    wave = np.arange(x[0]-25, x[-1]+25, 0.1)
    model = model_spec(wave, flux_unit=syn.units.FLAM)

    return (wave, model)

if __name__ == "__main__":
    # change these as needed
    LP = 'LP6'
    PID = '16830'

    # the directory where the data is stored
    data_dir = glob.glob('/grp/hst/cos2/new_TDSTAB_postgeo/DATA/calibrated/'+LP+'/'+PID+'/*x1d.fits')

    for file in data_dir:
        hdr0 = fits.getheader(file, 0)
        data = fits.getdata(file, 1)

        flux = np.mean(data['FLUX'][data['DQ_WGT'] != 0])
        wl   = np.mean(data['WAVELENGTH'][data['DQ_WGT'] != 0])
        wave, model = select_model(hdr0['TARGNAME'], data['WAVELENGTH'][data['DQ_WGT'] != 0])


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