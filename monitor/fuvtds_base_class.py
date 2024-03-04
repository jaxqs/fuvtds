from astropy.io import fits
import numpy as np
import glob
import os
from astropy.time import Time
import pandas as pd
import multiprocessing as mp
from itertools import repeat
from scipy.stats import binned_statistic

"""
This is the base class for the FUVTDS Monitor that will do all
the analysis need and provide the necessary component to the
monitor that will conduct all the plotting.

Please be aware, this will be a monster.
"""

__author__ = 'J. Hernandez' #Me! JAQ!

class FUVTDSBase:
    """
    This class will analyze and store the results necessary to
    conduct a FUVTDS monitor analysis.

    Attributes:
        infiles (array-like): files organized by date and cleaned, by segment.
        date_dec (array-like): the decimal years of all the files by chronological order, by segment.
        breakpoints (array-like): all the TDS breakpoints by fractional year.
        reftime (float.64): The decimal year of the reftime.
        nentries (int): number of datasets, by segment.
        rootnames (array-like): Rootnames of all input x1d files, by segment.
        nets (array-like): binned NET array for each x1d file, per segment.
        wls (array-like): binned  WAVELENGTH array for each x1d file, per segment.
        stdevs (array-like): binned standard deviation of the NET array binning.
        gratings (array-like): OPT_ELEM keyword for each x1d file, per segment.
        segments (array-like): SEGMENT keyword for each x1d file, per segment.
        nentries (int): the amount of entries of each x1d file, per segment.
        lps (array-like): the LIFE_ADJ keyword for each x1d file, per segment.
        targs (array-like): the TARGNAME keyword for each x1d file, per segment.
        ref (dict): Nested dictionary of NET, WAVELENGTH, and filename for each
                reference (first in time) dataset's cenwave.
    """

    def __init__(self, PIDs, reftime = 54952.0, 
                 breakpoints = [2010.2, 2011.2, 2011.75, 2012.0, 2012.8, 2013.8, 2015.5, 2019.0, 2020.6, 2022.0, 2023.2]):
        """
        Args:
            PIDs: the dat file that will store all the PIDs part of the
                FUVTDS Monitor programs.
            breakpoints: all the TDS breakpoints by fractional year.
        """
        files = self.parse_infiles(PIDs)
        self.breakpoints = np.array(breakpoints)
        self.reftime = Time(reftime, format="mjd").decimalyear
        net_len = self.get_hduinfo()
        self.bin_data()

        # scale between LPs here

        # after the scaling, get the reference data (first obs)
        self.get_refdata()
        scaled_net, scaled_std = self.calc_ratios()

# --------------------------------------------------------------------------------#
    def calc_ratios(self):
        """
        Scale the NET array and STDEV array of each input dataset to the reference
        dataset (first dataset of the same cenwave and segment).

        Note: Should I interpolate the NET values to the wavelength scale of its
        reference dataset (first in time)?
        """
        scaled_net = []
        scaled_std = []

        for i in range(self.nentries):

            # scale the NET array to its respective reference dataset
            ref_net = self.ref[self.cenwaves[i]][self.segments[i]]["net"]
            ref_net.astype(np.float64)
            ratio_net = self.nets[i]/ ref_net
            ratio_net = np.nan_to_num(ratio_net)
            scaled_net.append(ratio_net)

            # scale the STDEV array to its respective reference dataset
            # be sure to propagate errors correctly
            ref_std = self.ref[self.cenwaves[i]][self.segments[i]]["stdev"]
            ref_std.astype(np.float64)
            ratio_std = (np.sqrt((ref_std/ref_net)**2 + (self.stdevs[i]/self.nets[i])**2)) * ratio_net
            ratio_std = np.nan_to_num(ratio_std)
            scaled_std.append(ratio_std)
        
        return(np.array(scaled_net, dtype=object), np.array(scaled_std, dtype=object))

# --------------------------------------------------------------------------------#
    def bin_data(self):
        """
        Bin the net counts in each wavelength bin for each file and segment and
        calculcate the standard deviation.
        """
        wl_info_dict = {
            # G160M
            1533: {'FUVA':[1535.0, 1705.0, 5], 'FUVB': [1345.0, 1515.0, 5]},
            1577: {'FUVA':[1575.0, 1750.0, 5], 'FUVB': [1385.0, 1560.0, 5]},
            1611: {'FUVA':[1610.0, 1785.0, 5], 'FUVB': [1420.0, 1590.0, 5]},
            1623: {'FUVA':[1625.0, 1795.0, 5], 'FUVB': [1435.0, 1605.0, 5]},
            
            # G130M
            1222: {'FUVA':[1225.0, 1360.0, 5], 'FUVB': [1085.0, 1205.0, 20]},
            1055: {'FUVA':[1065.0, 1185.0, 20],'FUVB': [910.0, 1030.0, 60]},
            1096: {'FUVB':[950.0, 1070.0, 20]},
            1291: {'FUVA':[1290.0, 1430.0, 5], 'FUVB': [1140.0, 1200.0, 5]},
            1327: {'FUVA':[1325.0, 1470.0, 5], 'FUVB': [1170.0, 1315.0, 5]},

            # G140L 
            800 : {'FUVA':[920.0, 1800.0, 20]},
            1105: {'FUVA':[1140.0, 1800.0, 20]},
            1280: {'FUVA':[1280.0, 1800.0, 20],'FUVB': [1100.0, 1120.0, 20]}
            }
        
        wavelengths = []
        nets = []
        stdevs = []
        for i in range(self.nentries):

            wl_range = wl_info_dict[self.cenwaves[i]][self.segments[i]]
            min_wl = wl_range[0]
            max_wl = wl_range[1]
            binsize = wl_range[2]

            bins = np.arange(min_wl, max_wl, binsize)
            nbins = len(bins) - 1

            x_index = np.where((self.wls[i] >= min_wl) & (self.wls[i] <= max_wl))

            # Determine the mean and STD for each bin
            mean_net, edges, _ = binned_statistic(
                self.wls[i][x_index],
                self.nets[i][x_index],
                "mean", bins=bins
            )
            std_net = binned_statistic(
                self.wls[i][x_index],
                self.nets[i][x_index],
                np.std, bins=bins
            )[0]

            wavelengths.append(edges[:-1]+np.diff(edges)/2)
            nets.append(mean_net)
            stdevs.append(std_net)

        self.wls = np.array(wavelengths, dtype=object)
        self.nets = np.array(nets, dtype=object)
        self.stdevs = np.array(stdevs, dtype=object)
# --------------------------------------------------------------------------------#
    def get_refdata(self):
        """
        Determine the reference dataset (first in time) of each cenwave and
        store its NET, WAVELENGTH, and filename information in a dictionary.
        """

        ref_dict = {}
        for i in range(self.nentries):
            if self.cenwaves[i] not in ref_dict.keys():
                ref_dict[self.cenwaves[i]] = {}
            if self.segments[i] not in ref_dict[self.cenwaves[i]].keys():
                ref_dict[self.cenwaves[i]][self.segments[i]] = {}
                ref_dict[self.cenwaves[i]][self.segments[i]]["net"] = self.nets[i]
                ref_dict[self.cenwaves[i]][self.segments[i]]["wl"] = self.wls[i]
                ref_dict[self.cenwaves[i]][self.segments[i]]["stdev"] = self.stdevs[i]
                ref_dict[self.cenwaves[i]][self.segments[i]]["filename"] = self.infiles[i]
        
        self.ref = ref_dict

# --------------------------------------------------------------------------------#
    def get_hduinfo(self):
        """
        Get necessary information from the input files' HDU headers and data extensions.
        """

        nets = []
        wls = []
        cenwaves = []
        gratings = []
        segments = []
        nentries = []
        lps = []
        targets = []
        rootnames = []
        date_dec = []

        for i in range(len(self.infiles)):
            with fits.open(self.infiles[i], memmap=False) as hdulist:
                data = hdulist[1].data
                hdr0 = hdulist[0].header
                hdr1 = hdulist[1].header

                # if the x1d file is a single segment, try this
                if hdr0["segment"] != 'BOTH':
                    if hdr0["cenwave"] == 1230:
                        cenwaves.append(1280)
                    else:
                        cenwaves.append(hdr0["cenwave"])
                    gratings.append(hdr0["opt_elem"])
                    nets.append(data["net"][data["dq_wgt"] != 0])
                    wls.append(data["wavelength"][data["dq_wgt"] != 0])
                    segments.append(hdr0["segment"])
                    nentries.append(self.infiles[i])
                    lps.append(hdr0['LIFE_ADJ'])
                    targets.append(hdr0['TARGNAME'])
                    rootnames.append(hdr0['rootname'])
                    date_dec.append(Time(hdr1['EXPSTART'], format="mjd").decimalyear)

                # if the x1d file has BOTH segments, do this instead
                else:
                    FUV = ['FUVA', 'FUVB']
                    for j, seg in enumerate(FUV):
                        if hdr0["cenwave"] == 1230:
                            cenwaves.append(1280)
                        else:
                            cenwaves.append(hdr0["cenwave"])
                        gratings.append(hdr0["opt_elem"])
                        nets.append(data["net"][j][data["dq_wgt"][j] != 0])
                        wls.append(data["wavelength"][j][data["dq_wgt"][j] != 0])
                        segments.append(seg)
                        nentries.append(self.infiles[i])
                        lps.append(hdr0['LIFE_ADJ'])
                        targets.append(hdr0['TARGNAME'])
                        rootnames.append(hdr0['rootname'])
                        date_dec.append(Time(hdr1['EXPSTART'], format="mjd").decimalyear)

        self.nets = np.array(nets, dtype=object)
        self.wls = np.array(wls, dtype=object)
        self.cenwaves = np.array(cenwaves)
        self.gratings = np.array(gratings)
        self.segments = np.array(segments)
        self.nentries = len(nentries)
        self.lps = np.array(lps)
        self.targs = np.array(targets)
        self.rootnames = np.array(rootnames)
        self.date_dec = np.array(date_dec)
        self.infiles = np.array(nentries)
        return(len(self.nets)) #THIS IS A CHECK, NOT USED IN MONITOR

# --------------------------------------------------------------------------------#
    def parse_infiles(self, PIDs, COSMO = '/grp/hst/cos2/cosmo/', pattern='*x1d.fits*'):
       """
        Determine the list of all input files

        Args:
            PIDs: dat file of the PIDs 
            COSMO: the directory path that leads to the directory that stores the relevant files
            pattern: the x1d files
        Returns:
            x1d_paths: The complete paths to all the x1dfiles, in chronological order and
                    cleaned of all the datasets we do not want.
        """
       # read in dat file
       programs_df = pd.read_csv(PIDs, delim_whitespace=True)
       
       # change dat file into list of str of PID numbers
       all_programs = []
       for _, col_data in programs_df.items(): 
          all_programs += col_data.to_numpy(dtype=str).tolist()
       
       with mp.Pool(16) as pool:
           x1d_paths = pool.starmap(self._get_x1ds, zip(repeat(COSMO), all_programs, repeat(pattern)))
       x1d_paths = [x for sublist in x1d_paths for x in sublist] # flatten list
       pool.terminate()

       x1d_paths = np.array(x1d_paths)
       # filter out the data we do not want. ei WAVECAL, weird targets, and zero exptime
       # this takes a lot of time to run, perhaps a way to multithread this?
       bad_cenwaves = [1600, 1589, 1309]
       filter = [(fits.getval(x, "targname", 0) != 'LDS749B') & 
                         (fits.getval(x, "exptime", 1) != 0.0) &
                          (fits.getval(x, "EXPTYPE", 0) == 'EXTERNAL/SCI') &
                           (fits.getval(x, "cenwave",0) not in bad_cenwaves) for x in x1d_paths]
       x1d_paths = x1d_paths[filter]

       # order the files by mjd
       mjds = [fits.getval(x, "expstart", 1) for x in x1d_paths]
       order = np.argsort(mjds)
       self.infiles = x1d_paths[order]

       return(x1d_paths) #THIS RETURN IS NOT NEEDED FOR THE ACTUAL MONITOR. JSUT A CHECK
# --------------------------------------------------------------------------------#
    def _get_x1ds(self, COSMO, all_programs, pattern):
        """
        Obtain the full path to the x1d files from COSMO directory.
        Used solely for the parse_infiles(function).

        Args:
            COSMO: the directory in which the data is stored. By default,
                    this will be /grp/hst/cos2/cosmo.
            all_programs: the PIDs of all the programs used in the FUVTDS monitor
            pattern: the pattern of file we want, in this case x1d files.
        Returns:
            path_list: the path to the x1d file.
        """
        total_path = os.path.join(COSMO, all_programs, pattern)
        path_list = glob.glob(total_path)
        return(path_list)
    
    def _criteria(self, x1dfile):
        hdu = fits.open(x1dfile)

        bad_roots = {'G130M': ['lbxm04pbq', 'lbxm04pdq', 'lbxm04pfq', 'lbxm04phq','ldqj05xyq', 'ldqj08j1q', 
                           'ldv003d9q', 'ldv007p4q', 'ldv008o9q', 'ldv010ekq', 'ldv006lkq', 'ldqj05xuq', 
                           'ldqj08ixq', 'ldqj12e2q', 'ldqj56a3q', 'ldqj57trq', 'ldv003dbq', 'ldqj59jtq', 
                           'ldv007p6q', 'ldv008obq', 'ldv010eoq', 'lefe03gmq', 'ldqj05xwq', 'ldqj08izq', 
                           'ldqj12e4q', 'ldqj56a5q', 'ldqj57ttq', 'ldv003ddq', 'ldqj59jvq', 'ldv007p8q', 
                           'ldv010etq', 'ler107a7q', 'ldqj05yoq', 'ldqj08jdq', 'ldqj12e6q', 'ldqj56a7q', 
                           'ldqj57tvq', 'ldqj59jxq', 'ldv007pmq', 'ldv008orq', 'ldv010fvq', 'le5g07naq', 
                           'ler15bhaq', 'lf205ag7q', 'lefe03gjq', 'lbxmt3m3q', 'lbxm04p7q', 'lbxmt3mcq',
                           'lbxmt3m1q'], 
                'G140L': ['ldv007piq', 'ldv008onq', 'ldv010fpq', 'le5g07n4q', 'ler158owq', 'lf205ag2q',
                          'ldqj05ymq', 'ldqj08jbq', 'ldqj12eeq', 'ldqj56afq', 'ldqj57u3q', 'ldqj58hpq',
                          'ldqj59k5q', 'ldv007pkq', 'ldv008opq', 'ldv010ftq', 'le5g07n8q', 'lf205ag4q',
                          'ldqj05ykq', 'ldqj08j9q', 'ldqj12ecq', 'ldqj56adq', 'ldqj57u1q', 'ldqj58hmq',
                          'ldqj59k3q', 'ldv007paq', 'ldv010evq', 'lefe03grq', 'lbxmt1nzq', 'lbxmt3lwq',
                          'lbxmt3mfq'],
                'G160M': ['lbb917kfq', 'ldv006lqq', 'ldv007pcq', 'ler106dqq', 'lf2056ftq', 'ldqj05y0q', 
                          'ldqj08j3q', 'ldqj13e9q', 'ldqj12e8q', 'ldqj56a9q', 'ldqj57txq', 'ldqj58hgq', 
                          'ldv006lsq', 'ldqj59jzq', 'ldv007peq', 'ldv008ojq', 'ldv010flq', 'ler106dsq',
                          'lf2006liq', 'lf205bauq', 'lf2056g5q', 'ldqj05y2q', 'ldqj08j5q', 'ldqj13ebq', 
                          'ldqj12eaq', 'ldqj56abq', 'ldqj57tzq', 'ldqj58hjq', 'ldv006lvq', 'ldqj59k1q', 
                          'ldv007pgq', 'ldv008olq', 'ldv010fnq', 'ler106doq', 'ler106dwq', 'lf2006lqq', 
                          'lf205bb8q', 'lf2111zuq', 'ler106duq', 'ler106dkq', 'lbxm02agq', 'lbxm02bxq', 
                          'lf2111zoq', 'lf2006lmq', 'lbxm02ayq', 'lf205bb1q']} 

        def c1280_check(cenwave, fppos, expstart):
            good = False
            if (fppos == 3) & (cenwave != 1280): good = True
            elif (cenwave == 1280) & (fppos == 4) & (expstart < 56130.0): good = True
            elif (cenwave == 1280) & (fppos == 3) & (expstart > 56130.0): good = True
            return(good)
        
        criteria = {
            'exptime': hdu[1].header['exptime'] != 0,
            'bad_targs': hdu[0].header['targname'] not in ['WAVE', 'LDS749B'],
            'bad_items': hdu[0].header['rootname'] not in bad_roots[hdu[0].header['opt_elem']],
            'fppos_check': c1280_check(
                hdu[0].header['cenwave'],
                hdu[0].header['fppos'],
                hdu[1].header['expstart']) == True,
            'wl': len(hdu[1].data['wavelength']) != 0,
            'lp4': (hdu[0].header['opt_elem'] != 'G160M') | ((hdu[0].header['life_adj'] != 4) | (hdu[1].header['date-obs'] < '2022-10-01'))
        }

        ## potentially combine the analyze file inferstructure with the monitor
        ## so that this monitor will create a csv file as one of the output products?
        ## that way we don't have to update two different codes on any "bad" rootnames.
        ## this would also benefit from looking more at those Bad Items to see if there
        ## are cenwaves that are no longer used / certain criteria that can be coupled somewhere
        ## else instead of just explicitly putting the rootname as Bad.
        #return(cleaned)