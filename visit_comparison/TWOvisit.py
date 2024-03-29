from data_models import get_new_data
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import synphot as syn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
import os
import pandas as pd

def organize(data):
    diction = {}
    for d in data:
        if d['TARGNAME'] != 'WAVE':
            diction[d['CENWAVE'], d['LIFE_ADJ']] = d

    diction = pd.DataFrame(diction)
    return(diction)

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
        '1222': {'FUVA':[1225.0, 1360.0, 5], 'FUVB': [1085.0, 1205.0, 5]},
        '1055': {'FUVA':[1065.0, 1185.0, 20],'FUVB': [910.0, 1030.0, 60]},
        '1096': {'FUVB':[950.0, 1070.0, 20]},
        '1291': {'FUVA':[1290.0, 1430.0, 5], 'FUVB': [1135.0, 1275.0, 5]},
        '1327': {'FUVA':[1325.0, 1470.0, 5], 'FUVB': [1170.0, 1315.0, 5]},

        # G140L 
        '800' : {'FUVA':[920.0, 1800.0, 20]},
        '1105': {'FUVA':[1140.0, 1800.0, 20]},
        '1280': {'FUVA':[1280.0, 1800.0, 20],'FUVB': [1100.0, 1120.0, 5]}
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
        'GD71': 'gd71_mod_012.fits',
        'WD0308-565': 'wd0308_565_mod_007.fits'
    }
    model_spec = syn.SourceSpectrum.from_file(os.path.join(
        os.environ['PYSYN_CDBS'], 'calspec', targs[target]))
    
    wave = np.arange(x[0]-25, x[-1]+25, 0.1)
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

def plot_flux(data_ref, data_new, cenwave):
    flux_new, wl_new, bin_size_new = binned(
        cenwave,
        data_new['SEGMENT'],
        data_new['WAVELENGTH'][0][data_new['DQ_WGT'][0]!=0],
        data_new['FLUX'][0][data_new['DQ_WGT'][0]!=0]
    )

    flux_ref, wl_ref, _ = binned(
        cenwave,
        data_ref['SEGMENT'],
        data_ref['WAVELENGTH'][0][data_ref['DQ_WGT'][0]!=0],
        data_ref['FLUX'][0][data_ref['DQ_WGT'][0]!=0]
    )

    wave, model = select_model(data_new['TARGNAME'], data_new['WAVELENGTH'][0])

#plots
#PLOT BUNCH 1
    fig = plt.figure(tight_layout=True)
    gs  = gridspec.GridSpec(2,3)

    # PLOT 1 - COMPARISON OF THE TWO VISITS
    ax = fig.add_subplot(gs[0,:])
    ax.scatter(wl_new, (flux_new - flux_ref) / flux_ref, marker='v', c='g', label='Flux Difference')

    airglow_bins = airglow(data_ref['OPT_ELEM'], data_ref['CENWAVE'], float(bin_size_new))
    for a in airglow_bins:
        ax.axvspan(a[0], a[1], color='brown', alpha=0.1)

    ax.hlines(0, min(wave), max(wave), ls=':', color='k')
    ax.hlines([0.05, 0.02, -0.02, -0.05],min(wave),max(wave),ls='--',color='k')
    ax.set_title(f"{data_new['OPT_ELEM']}/{data_new['CENWAVE']}/{data_new['SEGMENT']}\
                 {data_new['TARGNAME']} {bin_size_new}Å-bin {data_ref['DATE-OBS']} ; {data_new['DATE-OBS']}")
    ax.set_ylim(-0.06,0.06)
    ax.set_xlim(data_new['WAVELENGTH'][0][0]-5, data_new['WAVELENGTH'][0][-1]+5)
    ax.set_ylabel('(Flux_New - Flux_Ref) / Flux_Ref')
    ax.set_xlabel('wavelength (Å)')
    ax.legend()

    # PLOT 2 - RESIDUALS OF THE TWO VISITS
    model_y_new  = np.array(np.interp(wl_new, wave, model))
    residual_new = (flux_new - model_y_new) / model_y_new

    model_y_ref  = np.array(np.interp(wl_ref, wave, model))
    residual_ref = (flux_ref - model_y_ref) / model_y_ref

    ax = fig.add_subplot(gs[1,:-1])
    ax.scatter(wl_new, residual_new, marker='o', c='blue', label=data_new['DATE-OBS'])
    ax.scatter(wl_ref, residual_ref, marker='x', c='r', label=data_ref['DATE-OBS'])
    ax.set_ylim(-0.20, 0.20)
    ax.set_xlim(data_new['WAVELENGTH'][0][0]-5, data_new['WAVELENGTH'][0][-1]+5)
    ax.set_ylabel('data / model - 1')
    ax.set_xlabel('wavelength (Å)')
    ax.legend()

    # PLOT 3 -- HISTOGRAM OF THE RESIDUALS
    ax = fig.add_subplot(gs[1:,-1])
    ax.hist(residual_new,bins=22,range=(-0.2, 0.2),color='b',histtype='step',hatch='/', label='New')
    ax.hist(residual_ref,bins=22,range=(-0.2, 0.2),color='r',histtype='step',hatch='.', label='Ref')
    ax.legend()

if __name__ == "__main__":
    # Change these parameters to what is specific to you
    ref = ['17328', '04']
    new = ['17328', '06']

    #Grab the data from COSMO
    data_ref = get_new_data(ref[0], ref[1])
    data_new = get_new_data(new[0], new[1])

    # organize the data by cenwave and transform into DataFrames
    data_ref = organize(data_ref)
    data_new = organize(data_new)

    pdf = PdfPages(f'output/{ref[0]}_visit{ref[1]}&{new[0]}_visit{new[1]}_comparison.pdf')
    for c, lp in data_new.columns:
        try:
            plot_flux(data_ref[c, lp], data_new[c, lp], c)
            pdf.savefig()
        except:
            continue
    pdf.close()