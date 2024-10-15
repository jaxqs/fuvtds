import numpy as np 
import synphot as syn
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import binned_statistic
import os
import pandas as pd
import glob
from astropy.io import fits
from matplotlib.backends.backend_pdf import PdfPages

def select_model(target, x):
    targs = {
        'GD71': 'gd71_mod_012.fits',
        'WD0308-565': 'wd0308_565_mod_007.fits',
        'WD1057+719': 'wd1057_719_mod_009.fits'
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

def binned(files, dir = '/grp/hst/cos2/cosmo/'):

    # cenwaves --> dictionary
    # 'cenwave': {SEGMENT_A, SEGMENT_B}
    # 'segment': [lower wavelength limit, upper wavelength limit, Angstrom-bin]
    cenwaves = {
        # G160M
        1533: {'FUVA':[1535.0, 1705.0, 5], 'FUVB': [1345.0, 1515.0, 5]},
        1577: {'FUVA':[1575.0, 1750.0, 5], 'FUVB': [1385.0, 1560.0, 5]},
        1611: {'FUVA':[1610.0, 1785.0, 5], 'FUVB': [1420.0, 1590.0, 5]},
        1623: {'FUVA':[1625.0, 1795.0, 5], 'FUVB': [1435.0, 1605.0, 5]},
        
        # G130M
        1222: {'FUVA':[1225.0, 1360.0, 5], 'FUVB': [1085.0, 1205.0, 5]},
        1055: {'FUVA':[1065.0, 1185.0, 20],'FUVB': [910.0, 1030.0, 60]},
        1096: {'FUVB':[950.0, 1070.0, 20]},
        1291: {'FUVA':[1290.0, 1430.0, 5], 'FUVB': [1135.0, 1275.0, 5]},
        1327: {'FUVA':[1325.0, 1470.0, 5], 'FUVB': [1170.0, 1315.0, 5]},

        # G140L 
        800 : {'FUVA':[920.0, 1800.0, 20]},
        1105: {'FUVA':[1140.0, 1800.0, 20]},
        1280: {'FUVA':[1280.0, 1800.0, 20],'FUVB': [1100.0, 1120.0, 5]}
        }


    tables = []

    files = glob.glob(f"{dir}{files['PID']}/*{files['visit']}*_x1d.fits*")

    for file in files:
        hdr0 = fits.getheader(file, 0)
        hdr1 = fits.getheader(file, 1)
        data = fits.getdata(file, 1)

        # skip bad datasets
        if (hdr1['exptime'] == 0) | (len(data['wavelength'][0]) == 0) | (hdr0['targname'] == 'WAVE'):
            continue

        for i, segment in enumerate(data['segment']):

            dqwgt = data['dq_wgt'][i]
            wl = data['wavelength'][i]
            flux = data['flux'][i]

            wl = wl[dqwgt != 0]
            flux = flux[dqwgt != 0]

            min_wl, max_wl, binsize = cenwaves[hdr0['cenwave']][segment]

            # the wavelength bin edges based on binsize
            bins = np.arange(min_wl, max_wl+1, binsize)
            x_index = np.where((wl >= min_wl) & (wl <= max_wl))

            # Determine the mean and STD for each bin
            mean_flux, edges, _ = binned_statistic(
                wl[x_index],
                flux[x_index],
                "mean", bins=bins)
            
            table = pd.DataFrame({
                'rootname': [hdr0['rootname']],
                'date-obs': [hdr1['date-obs']],
                'life_adj': [hdr0['life_adj']],
                'exptime': [hdr1['exptime']],
                'fppos': [hdr0['fppos']],
                'proposid': [hdr0['proposid']],
                'targname': [hdr0['targname']],
                'file_path': [file],
                'opt_elem': [hdr0['opt_elem']],
                'cenwave': [hdr0['cenwave']],
                'segment': [segment],
                'flux': [mean_flux],
                'wl_bins': [edges[:-1]+np.diff(edges)/2],
                'binsize': binsize
            })

            tables.append(table)
    

    # combine all the x1d dataframe tables into one
    tables = pd.concat(tables, ignore_index=True)

    return tables

def plot_compare(ref, new):
    pid_ref, visit_ref = ref['PID'], ref['visit']
    pid_new, visit_new = new['PID'], new['visit']
    ref = binned(ref)
    new = binned(new)

    combined = ref.merge(new,how='right', on=['opt_elem','cenwave', 'segment', 'life_adj', 'targname'], suffixes=['_ref', '_new'])

    combined = combined.dropna(ignore_index=True)

    pdf = PdfPages(f'output/{pid_ref}_visit{visit_ref}&{pid_new}_visit{visit_new}_comparison.pdf')

    for i in range(len(combined)):
        plot(combined.iloc[i])
        pdf.savefig()
    pdf.close()

def plot(df):

    wave, model = select_model(df['targname'], df['wl_bins_new'])

    #plots
    #PLOT BUNCH 1
    fig = plt.figure(tight_layout=True)
    gs  = gridspec.GridSpec(2,3)

    # PLOT 1 - COMPARISON OF THE TWO VISITS
    ax = fig.add_subplot(gs[0,:])
    ax.scatter(
        df['wl_bins_new'], 
        (df['flux_new'] - df['flux_ref']) / df['flux_ref'], 
        marker='v', c='g', 
        label='Flux Difference')

    airglow_bins = airglow(df['opt_elem'], df['cenwave'], float(df['binsize_new']))
    for a in airglow_bins:
        ax.axvspan(a[0], a[1], color='brown', alpha=0.1)

    ax.hlines(0, min(wave), max(wave), ls=':', color='k')
    ax.hlines([0.05, 0.02, -0.02, -0.05],min(wave),max(wave),ls='--',color='k')
    ax.set_title(f"{df['opt_elem']}/{df['cenwave']}/{df['segment']}   {df['targname']} {int(df['binsize_new'])}Å-bin {df['date-obs_ref']};{df['date-obs_new']}")
    ax.set_ylim(-0.06,0.06)
    ax.set_xlim(wave[0]-5, wave[-1]+5)
    ax.set_ylabel('(Flux_New - Flux_Ref) / Flux_Ref')
    ax.set_xlabel('wavelength (Å)')
    ax.legend()


    # PLOT 2 - RESIDUALS OF THE TWO VISITS
    model_y_new  = np.array(np.interp(df['wl_bins_new'], wave, model))
    residual_new = (df['flux_new'] - model_y_new) / model_y_new

    model_y_ref  = np.array(np.interp(df['wl_bins_ref'], wave, model))
    residual_ref = (df['flux_ref'] - model_y_ref) / model_y_ref

    ax = fig.add_subplot(gs[1,:-1])
    ax.scatter(df['wl_bins_new'], residual_new, marker='o', c='blue', label=df['date-obs_new'])
    ax.scatter(df['wl_bins_ref'], residual_ref, marker='x', c='r', label=df['date-obs_ref'])
    ax.set_ylim(-0.20, 0.20)
    ax.set_xlim(wave[0]-5, wave[-1]+5)
    ax.set_ylabel('data / model - 1')
    ax.set_xlabel('wavelength (Å)')
    ax.legend()

    # PLOT 3 -- HISTOGRAM OF THE RESIDUALS
    ax = fig.add_subplot(gs[1:,-1])
    ax.hist(residual_new,bins=22,range=(-0.2, 0.2),color='b',histtype='step',hatch='/', label='New')
    ax.hist(residual_ref,bins=22,range=(-0.2, 0.2),color='r',histtype='step',hatch='.', label='Ref')
    ax.legend()

if __name__ == "__main__":
    """
    """
    ref = {'PID': 17328,
           'visit': '9b'}
    new = {'PID': 17326,
           'visit': '9b'}
    
    plot_compare(ref, new)