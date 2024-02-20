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
        infiles (array-like): files organized by date and cleaned
        mjds (array-like): the MJDs of all the files by chronological order
        breakpoints (array-like): all the tDS breakpoints by fractional year.
        nentries (int): number of datasets, by segment
        rootnames (array-like): Rootnames of all input x1d files.
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

    Returns
    """

    def __init__(self, PIDs, breakpoints = [2010.2, 2011.2, 2011.75, 2012.0, 2012.8, 2013.8, 2015.5, 2019.0, 2020.6, 2022.0]):
        """
        Args:
            PIDs: the dat file that will store all the PIDs part of the
                FUVTDS Monitor programs.
            breakpoints: all the TDS breakpoints by fractional year.
        """
        files = self.parse_infiles(PIDs)
        self.breakpoints = np.array(breakpoints)
        self.rootnames = np.array([fits.getval(x, "rootname", 0) for x in self.infiles])
        self.get_hduinfo()
        self.bin_data()
        self.get_refdata()

# --------------------------------------------------------------------------------#
    def bin_data(self):
        """
        Bin the net counts in each wavelength bin for each file and segment and
        calculcate the standard deviation.
        Lifetime position (LP) and the target (targ) are also recorded.
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
            wavelength = []
            net = []
            stdev = []

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

            wavelength.append(edges[:-1]+np.diff(edges)/2)
            net.append(mean_net)
            stdev.append(std_net)
        wavelengths.append(wavelength)
        nets.append(net)
        stdevs.append(stdev)

        self.wls = np.array(wavelengths)
        self.nets = np.array(nets)
        self.stdevs = np.array(stdevs)
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
                ref_dict[self.cenwaves[i]]["net"] = self.nets[i]
                ref_dict[self.cenwaves[i]]["wl"] = self.wls[i]
                ref_dict[self.cenwaves[i]]["filename"] = self.infiles[i]
        
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

        for i in range(len(self.infiles)):
            with fits.open(self.infiles[i], memmap=False) as hdulist:
                data = hdulist[1].data
                hdr0 = hdulist[0].header

                # if the x1d file is a single segment, try this
                try:
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

                # if the x1d file has BOTH segments, do this instead
                except:
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

        self.nets = np.array(nets)
        self.wls = np.array(wls)
        self.cenwaves = np.array(cenwaves)
        self.gratings = np.array(gratings)
        self.segments = np.array(segments)
        self.nentries = len(nentries)
        self.lps = np.array(lps)
        self.targs = np.array(targets)

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
       filter = [(fits.getval(x, "targname", 0) != 'LDS749B') & 
                         (fits.getval(x, "exptime", 1) != 0.0) &
                          (fits.getval(x, "EXPTYPE", 0) == 'EXTERNAL/SCI') for x in x1d_paths]
       x1d_paths = x1d_paths[filter]

       # order the files by mjd
       mjds = [Time(fits.getval(x, "expstart", 1), format="mjd") for x in x1d_paths]
       order = np.argsort(mjds)

       self.mjds = np.array(mjds)[order]
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