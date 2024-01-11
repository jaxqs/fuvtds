"""
READ_MODELS.PY
AUTHOR: Jacqueline Hernandez
DATE: Jan 2024
LIVE LOCATION: https://github.com/jaxqs/fuvtds

PURPOSE:
    The purpose of the following code is to analyze the x1d files of a given Program ID (PID)
and given Lifetime Position (LP). This is done by taking the x1d files and evaluating based
on segment and cenwave to then bin the flux and wavelength arrays of each file in both (usually)
5-Angstrom bins and 1-Angstrom bins. These bins are compared to a model spectra found in CALSPEC
and the mean and standard deviation are noted on the plot based on the residuals.

    This code produces a 3-panel plot. The first plot is Flux vs Wavelength with both flux and
wavelength divided into bins and then compared to a model spectra of the target of the observation. 
The second plot is Residuals vs Wavelength for those same bins from that observation to 
model spectra comparison. The third plot is a histogram of the residuals from the second.
"""

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
        '1533': {'FUVA':[1535.0, 1705.0, 5, 1], 'FUVB': [1345.0, 1515.0, 5, 1]},
        '1577': {'FUVA':[1575.0, 1750.0, 5, 1], 'FUVB': [1385.0, 1560.0, 5, 1]},
        '1611': {'FUVA':[1610.0, 1785.0, 5, 1], 'FUVB': [1420.0, 1590.0, 5, 1]},
        '1623': {'FUVA':[1625.0, 1795.0, 5, 1], 'FUVB': [1435.0, 1605.0, 5, 1]},
        
        # G130M
        '1222': {'FUVA':[1225.0, 1360.0, 5, 1], 'FUVB': [1085.0, 1205.0, 20, 1]},
        '1055': {'FUVA':[1065.0, 1185.0, 20, 1],'FUVB': [910.0, 1030.0, 60, 1]},
        '1096': {'FUVB':[950.0, 1070.0, 20, 1]},
        '1291': {'FUVA':[1290.0, 1430.0, 5, 1], 'FUVB': [1135.0, 1275.0, 5, 1]},
        '1327': {'FUVA':[1325.0, 1470.0, 5, 1], 'FUVB': [1170.0, 1315.0, 5, 1]},

        # G140L 
        '800' : {'FUVA':[920.0, 1800.0, 20, 1]},
        '1105': {'FUVA':[1140.0, 1800.0, 20, 1]},
        '1280': {'FUVA':[1280.0, 1800.0, 20, 1],'FUVB': [1100.0, 1120.0, 20, 1]}
        }

    # only grab data within the established wl ranges in the dictionary
    x_index = np.where((x >= cenwaves[str(cenwave)][segment][0]) &
                          (x <= cenwaves[str(cenwave)][segment][1]))
    
    # mean - 5 Ang
    s_5, edges_5, _ = binned_statistic(x[x_index], y[x_index], statistic = 'mean', 
                                   bins = (cenwaves[str(cenwave)][segment][1] - 
                                           cenwaves[str(cenwave)][segment][0]) / 
                                           cenwaves[str(cenwave)][segment][2])
    # mean - 1 Ang
    s_1, edges_1, _ = binned_statistic(x[x_index], y[x_index], statistic = 'mean', 
                                   bins = (cenwaves[str(cenwave)][segment][1] - 
                                           cenwaves[str(cenwave)][segment][0]) / 
                                           cenwaves[str(cenwave)][segment][3])

    edges_5 = edges_5[:-1]+np.diff(edges_5)/2
    edges_1 = edges_1[:-1]+np.diff(edges_1)/2
    
    return(s_5, edges_5, s_1, edges_1, str(cenwaves[str(cenwave)][segment][2]))

def select_model(target, x):
    #target : the target the data has observed to
    #x      : the wavelength array

    # this dic can become more sophisticate for LPs once the ref files changes for them (if they do)
    targs = {
        'GD71': 'gd71_mod_011.fits',
        'WD0308-565': 'wd0308_565_mod_006.fits'
    }

    # grab the relavent file from calspec
    model_spec = syn.SourceSpectrum.from_file(os.path.join(
        os.environ['PYSYN_CDBS'], 'calspec', targs[target]))
    
    # the wavelength array the spectrum derived will be made from
    wave = np.arange(min(x)-100, max(x)+100, 0.1)
    model = model_spec(wave, flux_unit=syn.units.FLAM)

    return (wave, model)

def airglow(grating, cenwave, binsize):
    # grating: the grating that was observed. G140L have different airglow bins than the rest
    # cenwave: the cenwave that was observed. G140L/800 has diff airglow bins than the rest
    # binsize: Angstroms, the bin size width in angstroms to make the airglow bins
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
    # segment: the segment the data was observed in. should be FUVA or FUVB
    # hdr0   : the header of the fits file, necessary to get TARGNAME and other values from
    # data   : the data of the fits file, necessary to do the analysis
    # num    : OPTIONAL, if the segment is listed as BOTH, num aids in seperating by segment
    try:
        flux_5, wl_5, flux_1, wl_1, binsize = binned(
            hdr0['cenwave'],
            segment,
            data['wavelength'][num][data['dq_wgt'][num] != 0],
            data['flux'][num][data['dq_wgt'][num] != 0]
        )
    except:
        return
    
    # wavelength and model spectra of the given target that was observed
    wave, model = select_model(hdr0['TARGNAME'], wl_5)

    fig = plt.figure(tight_layout=True)
    gs  = gridspec.GridSpec(2,4)

    #plot 1 - binned scatter
    ax = fig.add_subplot(gs[0,:-1])
    ax.scatter(wl_1, flux_1, marker ='.', color='darkblue', label='TDSTAB',  edgecolor='black')
    ax.scatter(wl_5, flux_5, marker ='o', color='lightblue', label='Binned TDSTAB', edgecolor='black')
    ax.semilogy(wave, model, color='red', label='model')

    # establish airglow regions in the plot
    airglow_bins = airglow(hdr0['opt_elem'], hdr0['cenwave'], float(binsize))
    for a in airglow_bins:
        ax.axvspan(a[0], a[1], color='brown', alpha=0.1)

    ax.set_title(f"{hdr0['opt_elem']}/{hdr0['cenwave']}/{segment}")
    ax.set_xlim(wave[0]+75, wave[-1]-75)
    ax.set_ylim(min(flux_5) - 1e-13, max(flux_5) + 1e-13)
    ax.set_ylabel('flux')
    ax.set_xlabel('wavelength (Å)')
    ax.legend()

    # plot 2 - residuals
    model_y_5  = np.array(np.interp(wl_5, wave, model))
    residual_5 = (flux_5 - model_y_5) / model_y_5

    model_y_1  = np.array(np.interp(wl_1, wave, model))
    residual_1 = (flux_1 - model_y_1) / model_y_1

    ax = fig.add_subplot(gs[1,:-1])
    ax.scatter(wl_1, residual_1, marker='.', c='darkblue', edgecolor='black', label='Residual')
    ax.scatter(wl_5, residual_5, marker='o', c='lightblue', edgecolor='black', label='Binned Residual')
    ax.hlines(0, min(wave), max(wave), ls=':', color='k')
    ax.hlines([0.05, 0.02, -0.02, -0.05],min(wave),max(wave),ls='--',color='k')
    ax.text(0.55, 0.03, f"mean {np.round(np.mean(residual_5), 3)}, std {np.round(np.std(residual_5), 3)}",
            fontsize=10, color='tab:blue', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)
    ax.set_ylim(-0.20, 0.20)
    ax.set_xlim(wave[0]+75, wave[-1]-75)
    ax.set_ylabel('data / model - 1')
    ax.set_xlabel('wavelength (Å)')
    ax.legend()

    # plot 3 - histogram of the residuals
    ax = fig.add_subplot(gs[:,3])
    ax.hist(residual_5,bins=22,range=(-0.2, 0.2),color='k',histtype='step',hatch='/')
    pdf.savefig()

def plot_maker(file):
    # file: the fits file, will be opened and analysised 
    hdr0 = fits.getheader(file, 0)
    data = fits.getdata(file, 1)

    if (hdr0['targname'] == 'WAVE') | (len(data['wavelength']) == 0):
        return
    
    # check to see if SEGMENT is set as either FUVA or FUVB and if so, proceed as usual
    #if SEGMENT is set to BOTH, then the except statement will seperate the two
    try:
        sep_segs(hdr0['SEGMENT'], hdr0, data)
    except:
        segments = ['FUVA', 'FUVB']
        for i, s in enumerate(segments):
            sep_segs(s, hdr0, data, i)

if __name__ == "__main__":
    # change these as needed
    LP = 'LP6'
    PID = '17328'

    # the directory where the data is stored
    data_dir = glob.glob('/grp/hst/cos2/new_TDSTAB_postgeo/DATA/calibrated/'+LP+'/'+PID+'/*x1d.fits')

    # intiate where plots will be stored
    pdf = PdfPages(f'output/{LP}_{PID}_new_reffiles.pdf')

    for file in data_dir:
        plot_maker(file)
    pdf.close()