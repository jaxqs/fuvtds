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

    results = [[s_5, edges_5], [s_1, edges_1], str(cenwaves[str(cenwave)][segment][2])]
    
    return(results)

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

def plot(segment, hdr0, hdr1, og_data, new_data, num = None):

    # returns the following in a list:
    # [flux in 5 ang bins, wl in 5 ang bins], [flux in 1 ang bins, wl in 1 ang bins], binsize
    OG = binned(
        hdr0['cenwave'],
        segment,
        og_data['wavelength'][num][og_data['dq_wgt'][num] != 0],
        og_data['flux'][num][og_data['dq_wgt'][num] != 0]
    )
    NEW = binned(
        hdr0['cenwave'],
        segment,
        new_data['wavelength'][num][new_data['dq_wgt'][num] != 0],
        new_data['flux'][num][new_data['dq_wgt'][num] != 0]
    )

    # wavelength and model spectra of the given target that was observed
    wave, model = select_model(hdr0['TARGNAME'], OG[0][1])
    fig = plt.figure(tight_layout=True)
    gs  = gridspec.GridSpec(2,4)

    #plot 1 - binned scatter
    ax = fig.add_subplot(gs[0,:-1])
    ax.scatter(OG[1][1], OG[1][0], marker ='.', color='darkblue', label='Old TDSTAB',  edgecolor='black')
    ax.scatter(OG[0][1], OG[0][0], marker ='o', color='lightblue', label='Binned Old TDSTAB', edgecolor='black')
    ax.scatter(NEW[1][1], NEW[1][0], marker ='.', color='chocolate', label='New TDSTAB',  edgecolor='black')
    ax.scatter(NEW[0][1], NEW[0][0], marker ='o', color='darkorange', label='Binned New TDSTAB', edgecolor='black')
    ax.semilogy(wave, model, color='red', label='model')

    # establish airglow regions in the plot
    airglow_bins = airglow(hdr0['opt_elem'], hdr0['cenwave'], float(OG[2]))
    for a in airglow_bins:
        ax.axvspan(a[0], a[1], color='brown', alpha=0.1)

    ax.set_title(f"{hdr0['opt_elem']}/{hdr0['cenwave']}/{segment}\
                 {hdr0['rootname']} {hdr1['date-obs']}")
    ax.set_xlim(wave[0]+75, wave[-1]-75)
    ax.set_ylim(min(OG[0][0]) - 1e-13, max(OG[0][0]) + 1e-13)
    ax.set_ylabel('flux')
    ax.set_xlabel('wavelength (Å)')
    ax.legend(fontsize=6)

    # plot 2 - residuals
    ax = fig.add_subplot(gs[1,:-1])

    model_y_5  = np.array(np.interp(OG[0][1], wave, model))
    old_residual_5 = (OG[0][0]- model_y_5) / model_y_5
    model_y_1  = np.array(np.interp(OG[1][1], wave, model))
    residual_1 = (OG[1][0] - model_y_1) / model_y_1

    ax.scatter(OG[1][1], residual_1, marker='.', c='darkblue', edgecolor='black', label='Old Residual')
    ax.scatter(OG[0][1], old_residual_5, marker='o', c='lightblue', edgecolor='black', label='Binned Old Residual')
    ax.text(0.55, 0.03, f"mean {np.round(np.mean(old_residual_5), 3)}, std {np.round(np.std(old_residual_5), 3)}",
            fontsize=10, color='tab:blue', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    model_y_5  = np.array(np.interp(NEW[0][1], wave, model))
    new_residual_5 = (NEW[0][0]- model_y_5) / model_y_5
    model_y_1  = np.array(np.interp(NEW[1][1], wave, model))
    residual_1 = (NEW[1][0] - model_y_1) / model_y_1

    ax.scatter(NEW[1][1], residual_1, marker='.', c='chocolate', edgecolor='black', label='New Residual')
    ax.scatter(NEW[0][1], new_residual_5, marker='o', c='darkorange', edgecolor='black', label='Binned New Residual')
    ax.text(0.02, 0.03, f"mean {np.round(np.mean(new_residual_5), 3)}, std {np.round(np.std(new_residual_5), 3)}",
            fontsize=10, color='tab:orange', horizontalalignment='left', verticalalignment='bottom', transform=ax.transAxes)

    ax.hlines(0, min(wave), max(wave), ls=':', color='k')
    ax.hlines([0.05, 0.02, -0.02, -0.05],min(wave),max(wave),ls='--',color='k')
    ax.set_ylim(-0.20, 0.20)
    ax.set_xlim(wave[0]+75, wave[-1]-75)
    ax.set_ylabel('data / model - 1')
    ax.set_xlabel('wavelength (Å)')

    # plot 3 - histogram of the residuals
    ax = fig.add_subplot(gs[:,3])
    ax.hist(old_residual_5,bins=22,range=(-0.2, 0.2),color='darkblue',histtype='step',hatch='/')
    ax.hist(new_residual_5,bins=22,range=(-0.2, 0.2),color='darkorange',histtype='step',hatch='.')
    pdf.savefig()

    plt.close()

def sep_segs(old_file, new_file):
    # file: the fits file, will be opened and analysised 
    old_hdr0 = fits.getheader(old_file, 0)
    old_data = fits.getdata(old_file, 1)

    if (old_hdr0['targname'] == 'WAVE') | (len(old_data['wavelength']) == 0):
        return
    
    old_hdr1 = fits.getheader(old_file, 1)
    new_data = fits.getdata(new_file, 1)
    
    # check to see if SEGMENT is set as either FUVA or FUVB and if so, proceed as usual
    #if SEGMENT is set to BOTH, then the except statement will seperate the two
    try:
        plot(old_hdr0['SEGMENT'], old_hdr0, old_hdr1, old_data, new_data)
    except:
        segments = ['FUVA', 'FUVB']
        for i, s in enumerate(segments):
            plot(s, old_hdr0, old_hdr1, old_data, new_data, i)


if __name__ == "__main__":
    # change these as needed
    PIDS = ['14854', '15384', '15535', '15773', '16324', '16830', '17249', '17251', '17328', '17326']

    for PID in PIDS:
        # the directory where the data is stored
        og_tdstab_data = glob.glob(f'/grp/hst/cos2/fuv_tds_2024/data/old_calibrated/{PID}/*_x1d.fits')
        new_tdstab_data = glob.glob(f'/grp/hst/cos2/fuv_tds_2024/data/new_calibrated/{PID}/*_x1d.fits')

        # intiate where plots will be stored
        pdf = PdfPages(f'/grp/hst/cos2/fuv_tds_2024/output/{PID}_tdstab_comparison.pdf')

        for i, og in enumerate(og_tdstab_data):
            sep_segs(og, new_tdstab_data[i])
        pdf.close()