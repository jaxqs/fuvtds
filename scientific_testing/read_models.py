import numpy as np 
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
import synphot as syn
import matplotlib.gridspec as gridspec
import os
from scipy.stats import binned_statistic
from matplotlib.backends.backend_pdf import PdfPages

def binned(cenwave, segment, x, y): 

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
    s, edges, _ = binned_statistic(x[x_index], y[x_index], statistic = 'mean', 
                                   bins = (cenwaves[str(cenwave)][segment][1] - 
                                           cenwaves[str(cenwave)][segment][0]) / 
                                           cenwaves[str(cenwave)][segment][2])

    edges = edges[:-1]+np.diff(edges)/2
    
    return(s, edges, str(cenwaves[str(cenwave)][segment][2]))

def select_model(target, x):
    targs = {
        'GD71': 'gd71_mod_011.fits',
        'WD0308-565': 'wd0308_565_mod_006.fits'
    }
    model_spec = syn.SourceSpectrum.from_file(os.path.join(
        os.environ['PYSYN_CDBS'], 'calspec', targs[target]))
    
    wave = np.arange(min(x)-100, max(x)+100, 0.1)
    model = model_spec(wave, flux_unit=syn.units.FLAM)

    return (wave, model)

def airglow(grating, cenwave, binsize):
    if grating == 'G140L':
        if cenwave == 800:
            airglow_bins  = np.array([1205, 1225, 1305])
        else:
            airglow_bins = np.array([1210, 1280, 1310])
    else:
        airglow_bins = np.array([1212.5, 1217.5, 1302.5, 1302.5, 1307.5, 1352.5, 1357.5])

    airglow = []
    for a in airglow_bins:
         airglow.append(np.array([a-binsize/2., a+binsize/2.]))
    return(airglow)

def sep_segs(segment, hdr0, data, num = None):
    flux, wl, binsize = binned(
        hdr0['cenwave'],
        segment,
        data['wavelength'][num][data['dq_wgt'][num] != 0],
        data['flux'][num][data['dq_wgt'][num] != 0]
    )
    
    wave, model = select_model(hdr0['TARGNAME'], wl)

    fig = plt.figure(tight_layout=True)
    gs  = gridspec.GridSpec(2,4)

    #plot 1 - binned scatter
    ax = fig.add_subplot(gs[0,:-1])
    ax.scatter(wl, flux, marker ='o', color='lightblue', label='Binned TDSTAB binned', edgecolor='black')
    ax.semilogy(wave, model, color='red', label='model')

    airglow_bins = airglow(hdr0['opt_elem'], hdr0['cenwave'], float(binsize))
    for a in airglow_bins:
        ax.axvspan(a[0], a[1], color='brown', alpha=0.1)

    ax.set_title(f"{hdr0['opt_elem']}/{hdr0['cenwave']}/{segment}")
    ax.set_xlim(wave[0]+75, wave[-1]-75)
    ax.set_ylabel('flux')
    ax.set_xlabel('wavelength (Å)')
    ax.legend()

    # plot 2 - residuals
    model_y  = np.array(np.interp(wl, wave, model))
    residual = (flux - model_y) / model_y

    ax = fig.add_subplot(gs[1,:-1])
    ax.scatter(wl, residual, marker='o', c='lightblue', edgecolor='black', label='Residual')
    ax.hlines(0, min(wave), max(wave), ls=':', color='k')
    ax.hlines([0.05, 0.02, -0.02, -0.05],min(wave),max(wave),ls='--',color='k')
    ax.text(0.55, 0.03, f"mean {np.round(np.mean(residual), 3)}, std {np.round(np.std(residual), 3)}",
            fontsize=10, color='tab:blue', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    ax.set_ylim(-0.20, 0.20)
    ax.set_xlim(wave[0]+75, wave[-1]-75)
    ax.set_ylabel('data / model - 1')
    ax.set_xlabel('wavelength (Å)')
    ax.legend()

    # plot 3 - histogram
    ax = fig.add_subplot(gs[:,3])
    ax.hist(residual,bins=22,range=(-0.2, 0.2),color='k',histtype='step',hatch='/')

def plot_maker(file):
    hdr0 = fits.getheader(file, 0)
    data = fits.getdata(file, 1)

    if (hdr0['targname'] == 'WAVE') | (len(data['wavelength']) == 0):
        return
    try:
        sep_segs(hdr0['SEGMENT'], hdr0, data)
        pdf.savefig()
    except:
        segments = ['FUVA', 'FUVB']
        for i, s in enumerate(segments):
            sep_segs(s, hdr0, data, i)
            pdf.savefig()

if __name__ == "__main__":
    # change these as needed
    LP = 'LP6'
    PID = '17249'

    # the directory where the data is stored
    data_dir = glob.glob('/grp/hst/cos2/new_TDSTAB_postgeo/DATA/calibrated/'+LP+'/'+PID+'/*x1d.fits')

    pdf = PdfPages(f'output/{LP}_{PID}_new_tdstab.pdf')

    for file in data_dir:
        plot_maker(file)
    pdf.close()