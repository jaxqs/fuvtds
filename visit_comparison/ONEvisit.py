from data_models import get_new_data
from matplotlib.backends.backend_pdf import PdfPages

import numpy as np
import synphot as syn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
import os

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
        'GD71': 'gd71_mod_011.fits',
        'WD0308-565': 'wd0308_565_mod_006.fits'
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

def plot_flux(data, segment, num=0):
     
    flux, wl, bin_size = binned(
        data['CENWAVE'],
        segment,
        data['WAVELENGTH'][num][data['DQ_WGT'][num]!=0],
        data['FLUX'][num][data['DQ_WGT'][num]!=0],
    )

    wave, model = select_model(data['TARGNAME'], data['WAVELENGTH'][num])

#plots
#PLOT BUNCH 1
    fig = plt.figure(tight_layout=True)
    gs  = gridspec.GridSpec(2,3)

    # PLOT 1 - COMPARE MODEL TO OBSERVATIONS
    ax = fig.add_subplot(gs[0,:])
    ax.scatter(wl, flux, marker='v', c='r', label='Observed')
    ax.plot(wave, model, color='blue', label='Model')

    airglow_bins = airglow(data['OPT_ELEM'], data['CENWAVE'], float(bin_size))
    for a in airglow_bins:
        ax.axvspan(a[0], a[1], color='brown', alpha=0.1)

    ax.set_title(f"{data['OPT_ELEM']}/{data['CENWAVE']}/{segment}\
                 {data['TARGNAME']} {bin_size}Å-bin {data['DATE-OBS']}")
    ax.set_xlim(data['WAVELENGTH'][num][0]-5, data['WAVELENGTH'][num][-1]+5)
    ax.set_ylabel('flux')
    ax.set_xlabel('wavelength (Å)')
    ax.legend()
    # PLOT 2 - RESIDUALS OF THE PLOT ABOVE
    model_y  = np.array(np.interp(wl, wave, model))
    residual = (flux - model_y) / model_y

    ax = fig.add_subplot(gs[1,:-1])
    ax.scatter(wl, residual, marker='v', c='r', label='Residual')
    ax.hlines(0, min(wave), max(wave), ls=':', color='k')
    ax.hlines([0.05, 0.02, -0.02, -0.05],min(wave),max(wave),ls='--',color='k')
    ax.set_ylim(-0.20, 0.20)
    ax.set_xlim(wave[0]+75, wave[-1]-75)
    ax.set_ylabel('data / model - 1')
    ax.set_xlabel('wavelength (Å)')
    ax.legend()

    # PLOT 3 -- HISTOGRAM OF THE RESIDUALS
    ax = fig.add_subplot(gs[1:,-1])
    ax.hist(residual,bins=22,range=(-0.2, 0.2),color='k',histtype='step',hatch='/')
    pdf.savefig()


def plot_net(data, segment, num=0):
    if (data['TARGNAME'] == 'WAVE'):
        return
    net, wl, bin_size = binned(
        data['CENWAVE'],
        segment,
        data['WAVELENGTH'][num][data['DQ_WGT'][num]!=0],
        data['NET'][num][data['DQ_WGT'][num]!=0]
    )
    bkg, _, _ = binned(
        data['CENWAVE'],
        segment,
        data['WAVELENGTH'][num][data['DQ_WGT'][num]!=0],
        data['BACKGROUND'][num][data['DQ_WGT'][num]!=0],
    )


# PLOTS BUNCH 2
    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 1)

    # PLOT 1 - NET and BACKGROUND vs WAVELENGTH
    ax = fig.add_subplot(gs[0])
    ax.scatter(wl, net, marker='v', color='red', label='Net Counts')
    ax.scatter(wl, bkg, marker='v', color='blue', label=' Background')
    ax.set_ylabel('Net counts')
    ax.set_xlabel('Wavelength (Å)')
    ax.set_title(f"{data['OPT_ELEM']}/{data['CENWAVE']}/{segment}\
                  {data['TARGNAME']} {bin_size}Å-bin {data['DATE-OBS']}")
    ax.legend()

    # PLOT 2 - BACKGROUND
    ax = fig.add_subplot(gs[1])
    ax.scatter(wl, bkg, marker='v', color='blue', label='Background')
    ax.set_ylabel('Net counts')
    ax.set_xlabel('Wavelength (Å)')
    ax.legend()
    pdf.savefig()

def seperate_segs(data, segment, func):
    if (data['TARGNAME'] == 'WAVE'):
        return
    
    try:
        func(data, segment)
    except:
        segments = ['FUVA', 'FUVB']
        for i, s in enumerate(segments):
            try: 
                func(data, s, i)
            except: pass

if __name__ == "__main__":
    # Change these parameters to what is specific to you
    PID = '17328'
    visit = '5b'

    data = get_new_data(PID, visit)

    pdf = PdfPages(f'output/{PID}_visit{visit}_flux.pdf')
    for d in data:
        seperate_segs(d, d['SEGMENT'], plot_flux)
    pdf.close()

    pdf = PdfPages(f'output/{PID}_visit{visit}_net.pdf')
    for d in data:
        seperate_segs(d, d['SEGMENT'], plot_net)
    pdf.close()