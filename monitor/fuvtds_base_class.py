from astropy.io import fits
import numpy as np
import glob
import os
from astropy.time import Time
import pandas as pd
import multiprocessing as mp
from itertools import repeat
from scipy.stats import binned_statistic
from tqdm.contrib.concurrent import process_map

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


        ADD SMALL AND LARGE DIFFERENCES HERE
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
                 breakpoints = [2010.2, 2011.2, 2011.75, 2012.0, 2012.8, 2013.8, 2015.5, 2019.0, 2020.6, 2022.0, 2023.2],
                 inventory='inventory_test.csv'):
        """
        Args:
            PIDs: the dat file that will store all the PIDs part of the
                FUVTDS Monitor programs.
            breakpoints: all the TDS breakpoints by fractional year.
        """
        self.cenwaves = [1533, 1577, 1623, 1291, 1327, 1222, 1105, 1280, 800, 1055, 1096]
        self.parse_infiles(PIDs, inventory)
        self.breakpoints = np.array(breakpoints)
        self.reftime = Time(reftime, format="mjd").decimalyear
        data_dictionary = self.get_hduinfo(inventory)
        small_dic = self.bin_data(data_dictionary, 'small')
        large_dic = self.bin_data(data_dictionary, 'large')

        # scale between LPs here
        self.scale_lps(large_dic)
        #if (6 in self.lps) & (4 in self.lps):
        #    self.scale_lp6_data()


        # after the scaling, get the reference data (first obs)
        #self.get_refdata()
        #scaled_net, scaled_std = self.calc_ratios()
            
# --------------------------------------------------------------------------------#
    def scale_lps(self, dictionary):
        """
        explain here <3
        """

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return (idx)
        
        def calc_error(i, dic_set, first_index, second_index):
            first_error = dic_set['stdev'][first_index[-1], i]
            first_data  = dic_set['binned_net'][first_index[-1], i]

            second_error = dic_set['stdev'][second_index[0], i]
            second_data  = dic_set['binned_net'][second_index[0], i]

            scale_factor_stdev = np.sqrt(
                (first_error/second_data)**2 +
                (first_data/(second_data**2)*second_error)**2
                )
            stdev = np.sqrt(
                dic_set['scale_factor'][second_index, i]**2 * 
                dic_set['stdev'][second_index, i]**2 +
                dic_set['binned_net'][second_index, i]**2 *
                scale_factor_stdev
            )
            return (stdev)

        for cenwave in dictionary:
            for segment in dictionary[cenwave]:
                if (6 in dictionary[cenwave][segment]['lp']) & (4 in dictionary[cenwave][segment]['lp']):
                    
                    lp6_indx_wd308 = np.where((dictionary[cenwave][segment]['lp'] == 6) &
                                            (dictionary[cenwave][segment]['target'] == 'WD0308-565'))
                    lp6_indx_gd71 = np.where((dictionary[cenwave][segment]['lp'] == 6) &
                                            (dictionary[cenwave][segment]['target'] == 'GD71'))
                    
                    lp4_indx_wd308 = np.where((dictionary[cenwave][segment]['lp'] == 4) &
                                            (dictionary[cenwave][segment]['target'] == 'WD0308-565'))
                    lp4_indx_gd71 = np.where((dictionary[cenwave][segment]['lp'] == 4) &
                                            (dictionary[cenwave][segment]['target'] == 'GD71'))
                    
                    if (cenwave > 1500) & (segment == 'FUVA'):
                        lp4_indx = lp4_indx_gd71
                        lp6_indx = lp6_indx_gd71
                    else: 
                        lp4_indx = lp4_indx_wd308
                        lp6_indx = lp6_indx_wd308

                    lp4_indx = lp4_indx[0]
                    lp6_indx = lp6_indx[0]

                    #Find the connection visit by closest date. Only a safe assumption for LP5 and 
                    #LP6 where the visits happened on the same day.
                    a = find_nearest(dictionary[cenwave][segment]['date'][lp4_indx], 
                                     dictionary[cenwave][segment]['date'][lp6_indx[0]])

                    for i, _ in enumerate(dictionary[cenwave][segment]['binned_wl']):
                        #Scale LP6 data
                        dictionary[cenwave][segment]['scale_factor'][lp6_indx, i] = (
                            dictionary[cenwave][segment]['binned_net'][lp4_indx[a], i] /
                            dictionary[cenwave][segment]['binned_net'][lp6_indx[0], i]
                        )
                        dictionary[cenwave][segment]['scaled_net'][lp6_indx, i] = (
                            dictionary[cenwave][segment]['scaled_net'][lp6_indx, i] *
                            dictionary[cenwave][segment]['scale_factor'][lp6_indx, i]
                        )
                        
                        # calculate error
                        dictionary[cenwave][segment]['scaled_stdev'][lp6_indx, i] = calc_error(
                            i, dictionary[cenwave][segment], lp4_indx, lp6_indx
                        )
                    print(f"+++ Scaling LP6 to LP4 using data from datasets: {dictionary[cenwave][segment]['infiles'][lp4_indx[a]]} {dictionary[cenwave][segment]['infiles'][lp6_indx[0]]}")


                if (5 in dictionary[cenwave][segment]['lp']) & (4 in dictionary[cenwave][segment]['lp']):

                    lp5_indx_wd308 = np.where((dictionary[cenwave][segment]['lp'] == 5) &
                                            (dictionary[cenwave][segment]['target'] == 'WD0308-565'))
                    lp5_indx_gd71 = np.where((dictionary[cenwave][segment]['lp'] == 5) &
                                            (dictionary[cenwave][segment]['target'] == 'GD71'))
                    
                    lp4_indx_wd308 = np.where((dictionary[cenwave][segment]['lp'] == 4) &
                                            (dictionary[cenwave][segment]['target'] == 'WD0308-565'))
                    lp4_indx_gd71 = np.where((dictionary[cenwave][segment]['lp'] == 4) &
                                            (dictionary[cenwave][segment]['target'] == 'GD71'))
                    
                    if (cenwave > 1500) & (segment == 'FUVA'):
                        lp4_indx = lp4_indx_gd71
                        lp5_indx = lp5_indx_gd71
                    else: 
                        lp4_indx = lp4_indx_wd308
                        lp5_indx = lp5_indx_wd308

                    lp4_indx = lp4_indx[0]
                    lp5_indx = lp5_indx[0]

                    a = find_nearest(dictionary[cenwave][segment]['date'][lp4_indx], 
                                     dictionary[cenwave][segment]['date'][lp5_indx[0]])
                    
                    for i, _ in enumerate(dictionary[cenwave][segment]['binned_wl']):
                        #Scale LP5 data
                        dictionary[cenwave][segment]['scale_factor'][lp5_indx, i] = (
                            dictionary[cenwave][segment]['binned_net'][lp4_indx[a], i] /
                            dictionary[cenwave][segment]['binned_net'][lp5_indx[0], i]
                        )
                        dictionary[cenwave][segment]['scaled_net'][lp5_indx, i] = (
                            dictionary[cenwave][segment]['scaled_net'][lp5_indx, i] *
                            dictionary[cenwave][segment]['scale_factor'][lp5_indx, i]
                        )
                        
                        # calculate error
                        dictionary[cenwave][segment]['scaled_stdev'][lp5_indx, i] = calc_error(
                            i, dictionary[cenwave][segment], lp4_indx, lp5_indx
                        )
                    print(f"+++ Scaling LP5 to LP4 using data from datasets: {dictionary[cenwave][segment]['infiles'][lp4_indx[a]]} {dictionary[cenwave][segment]['infiles'][lp5_indx[0]]}")
                


                if (4 in dictionary[cenwave][segment]['lp']) & (3 in dictionary[cenwave][segment]['lp']):
                    lp4_indx_wd308 = np.where((dictionary[cenwave][segment]['lp'] == 4) &
                                            (dictionary[cenwave][segment]['target'] == 'WD0308-565'))
                    lp4_indx_gd71 = np.where((dictionary[cenwave][segment]['lp'] == 4) &
                                            (dictionary[cenwave][segment]['target'] == 'GD71'))
                    
                    lp3_indx1_wd308 = np.where((dictionary[cenwave][segment]['lp'] == 3) &
                                            (dictionary[cenwave][segment]['target'] == 'WD0308-565') &
                                            (dictionary[cenwave][segment]['date'] < 2019.0))
                    lp3_indx1_gd71 = np.where((dictionary[cenwave][segment]['lp'] == 3) &
                                            (dictionary[cenwave][segment]['target'] == 'GD71')&
                                            (dictionary[cenwave][segment]['date'] < 2019.0))
                    
                    lp3_indx2_wd308 = np.where((dictionary[cenwave][segment]['lp'] == 3) &
                                            (dictionary[cenwave][segment]['target'] == 'WD0308-565') &
                                            (dictionary[cenwave][segment]['date'] > 2019.0))
                    lp3_indx2_gd71 = np.where((dictionary[cenwave][segment]['lp'] == 3) &
                                            (dictionary[cenwave][segment]['target'] == 'GD71')&
                                            (dictionary[cenwave][segment]['date'] > 2019.0))
                    

                    if (cenwave > 1500) & (segment == 'FUVA'):
                        lp3_indx1 = lp3_indx1_gd71
                        lp3_indx2 = lp3_indx2_gd71
                        lp4_indx  = lp4_indx_gd71
                    else: 
                        lp3_indx1 = lp3_indx1_wd308
                        lp3_indx2 = lp3_indx2_wd308
                        lp4_indx  = lp4_indx_wd308

                    lp3_indx1 = lp3_indx1[0] #LP3 indicies before LP3->LP4->LP3.
                    lp3_indx2 = lp3_indx2[0] #LP3 indicies after LP3->LP4->LP3.
                    lp4_indx  = np.concatenate((lp4_indx[0], lp3_indx2)) #lp4_indx needs to contain the post 2021 lp3_indx

                    for i, _ in enumerate(dictionary[cenwave][segment]['binned_wl']):
                        #Scale LP4 and LP3 after LP4->LP3
                        if cenwave != 800:
                            dictionary[cenwave][segment]['scale_factor'][lp4_indx, i] = (
                                dictionary[cenwave][segment]['binned_net'][lp4_indx[0]-1, i] /
                                dictionary[cenwave][segment]['binned_net'][lp4_indx[0], i]
                            )
                            dictionary[cenwave][segment]['scaled_net'][lp4_indx, i] = (
                                dictionary[cenwave][segment]['scaled_net'][lp4_indx, i] *
                                dictionary[cenwave][segment]['scale_factor'][lp4_indx, i]
                            )
                            # calculate error
                            dictionary[cenwave][segment]['scaled_stdev'][lp4_indx, i] = calc_error(
                                i, dictionary[cenwave][segment], lp3_indx1, lp4_indx
                            )
                            print(f"+++ Scaling LP4 to LP3 using data from datasets: {dictionary[cenwave][segment]['infiles'][lp4_indx[0]-1]} {dictionary[cenwave][segment]['infiles'][lp4_indx[0]]}")

                            # Scale LP3 after LP4 -> LP3
                            if len(lp3_indx2) > 0:
                                dictionary[cenwave][segment]['scale_factor'][lp3_indx2, i] = (dictionary[cenwave][segment]['binned_net'][lp4_indx[0], i] / dictionary[cenwave][segment]['binned_net'][lp4_indx[0]-1, i])
                                dictionary[cenwave][segment]['scaled_net'][lp3_indx2, i] = dictionary[cenwave][segment]['scaled_net'][lp3_indx2, i] * dictionary[cenwave][segment]['scale_factor'][lp3_indx2, i]

                                # calculate error
                                dictionary[cenwave][segment]['scaled_stdev'][lp3_indx2, i] = calc_error(
                                    i, dictionary[cenwave][segment], lp4_indx, lp3_indx2
                                )
                                print ('+++ Scaling LP3 to LP4 using data from datasets: ', dictionary[cenwave][segment]['infiles'][lp4_indx[0]], dictionary[cenwave][segment]['infiles'][lp4_indx[0]-1])
                        elif cenwave == 800:
                            dictionary[cenwave][segment]['scale_factor'][lp3_indx2, i] = (dictionary[cenwave][segment]['binned_net'][lp3_indx2[0]+1, i] / dictionary[cenwave][segment]['binned_net'][lp3_indx2[0], i])
                            dictionary[cenwave][segment]['scaled_net'][lp3_indx2, i] = dictionary[cenwave][segment]['scaled_net'][lp3_indx2, i] * dictionary[cenwave][segment]['scale_factor'][lp3_indx2, i]
                            # calculate error
                            dictionary[cenwave][segment]['scaled_stdev'][lp3_indx2, i] = calc_error(
                                i, dictionary[cenwave][segment], lp4_indx, lp3_indx2
                            )

                            print ('+++ Scaling LP3 to LP4 using data from datasets: ', dictionary[cenwave][segment]['infiles'][lp3_indx2[0]+1], dictionary[cenwave][segment]['infiles'][lp3_indx2[0]])

                #if (3 in dictionary[cenwave][segment]['lp']) & (2 in dictionary[cenwave][segment]['lp']): 

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
    def bin_data(self, data_dic, size):
        """
        Bin the net counts in each wavelength bin for each file and segment and
        calculcate the standard deviation.
        """
        wl_info_dict = {
            'small':{
                # G160M
                1533: {'FUVA':[1535.0, 1705.0, 5], 'FUVB': [1345.0, 1515.0, 5]},
                1577: {'FUVA':[1575.0, 1750.0, 5], 'FUVB': [1385.0, 1560.0, 5]},
                1623: {'FUVA':[1625.0, 1795.0, 5], 'FUVB': [1435.0, 1605.0, 5]},
                
                # G130M
                1222: {'FUVA':[1225.0, 1360.0, 5], 'FUVB': [1085.0, 1205.0, 20]},
                1055: {'FUVA':[1065.0, 1185.0, 20],'FUVB': [910.0, 1030.0, 60]},
                1096: {'FUVB':[950.0, 1070.0, 20]},
                1291: {'FUVA':[1290.0, 1430.0, 5], 'FUVB': [1135.0, 1275.0, 5]},
                1327: {'FUVA':[1325.0, 1470.0, 5], 'FUVB': [1170.0, 1315.0, 5]},

                # G140L 
                800 : {'FUVA':[1115.0, 1915.0, 20]},
                1105: {'FUVA':[1140.0, 2000.0, 20]},
                1280: {'FUVA':[1280.0, 2000.0, 20],'FUVB': [1100.0, 1120.0, 20]}
                },
            'large':{
                # G160M
                1533:{'FUVA':[1535, 1705, 170], 'FUVB':[1345, 1515, 170]},
                1577:{'FUVA':[1575, 1750, 175], 'FUVB':[1395, 1565, 170]},
                1623:{'FUVA':[1625, 1745, 120], 'FUVB':[1435, 1605, 170]},

                # G130M
                1222:{'FUVA':[1225, 1360, 135], 'FUVB':[1085, 1205, 120]},
                1055:{'FUVA':[1065, 1185, 120], 'FUVB':[910, 1030, 120]},
                1096:{'FUVB':[950, 1070, 120]},
                1291:{'FUVA':[1290, 1430, 140], 'FUVB':[1140, 1200, 60]},
                1327:{'FUVA':[1325, 1470, 145], 'FUVB':[1230, 1315, 85]},

                # G140L
                800:{'FUVA':[915, 1915, 1000]},
                1105:{'FUVA':[1300, 2000, 700]},
                1280:{'FUVA':[1280, 2000, 720], 'FUVB':[1100, 1120, 20]}
                }
            }
        
        dictionary = {}

        for cenwave in self.cenwaves:
            if cenwave not in dictionary.keys():
                dictionary[cenwave] = {}

            for segment in data_dic[cenwave]:
                if segment not in dictionary[cenwave].keys():
                    dictionary[cenwave][segment] = {
                        'binned_net': [],
                        'binned_wl' : [],
                        'stdev': [],
                        'grating': [],
                        'lp': [],
                        'target': [],
                        'rootname': [],
                        'date': [],
                        'infiles': []
                    }
                
                wl_range = wl_info_dict[size][cenwave][segment]
                min_wl = wl_range[0]
                max_wl = wl_range[1]
                binsize = wl_range[2]

                bins = np.arange(min_wl, max_wl, binsize)

                for i, wl in enumerate(data_dic[cenwave][segment]['wavelength']):
                    x_index = np.where((wl >= min_wl) & (wl <= max_wl))

                    # Determine the mean and STD for each bin
                    mean_net, edges, _ = binned_statistic(
                        wl[x_index],
                        data_dic[cenwave][segment]['net'][i][x_index],
                        "mean", bins=bins
                    )
                    std_net = binned_statistic(
                        wl[x_index],
                        data_dic[cenwave][segment]['net'][i][x_index],
                        np.std, bins=bins
                    )[0]

                    dictionary[cenwave][segment]['binned_net'].append(mean_net)
                    dictionary[cenwave][segment]['stdev'].append(std_net)

                dictionary[cenwave][segment]['binned_wl'] = np.array(edges[:-1]+np.diff(edges)/2)
                dictionary[cenwave][segment]['grating'] = data_dic[cenwave][segment]['grating']
                dictionary[cenwave][segment]['lp'] = data_dic[cenwave][segment]['lp']
                dictionary[cenwave][segment]['target'] = data_dic[cenwave][segment]['target']
                dictionary[cenwave][segment]['rootname'] = data_dic[cenwave][segment]['rootname']
                dictionary[cenwave][segment]['date'] = data_dic[cenwave][segment]['date']
                dictionary[cenwave][segment]['infiles'] = data_dic[cenwave][segment]['infiles']

        # reformat + add scaled components
        for cenwave in self.cenwaves:
            for segment in dictionary[cenwave]:
                dictionary[cenwave][segment]['binned_net'] = np.reshape(
                    dictionary[cenwave][segment]['binned_net'],
                    (len(dictionary[cenwave][segment]['infiles']),
                    len(dictionary[cenwave][segment]['binned_wl']))) # [date, wl_bin]
                dictionary[cenwave][segment]['scaled_net'] = dictionary[cenwave][segment]['binned_net']

                dictionary[cenwave][segment]['stdev'] = np.reshape(
                    dictionary[cenwave][segment]['stdev'],
                    (len(dictionary[cenwave][segment]['infiles']),
                    len(dictionary[cenwave][segment]['binned_wl']))) # [date, wl_bin]
                dictionary[cenwave][segment]['scaled_stdev'] = dictionary[cenwave][segment]['stdev']

                dictionary[cenwave][segment]['scale_factor'] = np.copy(dictionary[cenwave][segment]['binned_net'])*0.0+1.0
        return (dictionary)

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
    def get_hduinfo(self, csv_file):

        inventory = pd.read_csv(csv_file)

        data_dic = {}

        for cenwave in self.cenwaves:
            if cenwave not in data_dic.keys():
                data_dic[cenwave] = {}
            
            files = np.array(inventory['file_path'][(inventory['cenwave'] == cenwave)]).flatten()
            for file in files:
                with fits.open(file, memmap=False) as hdulist:
                    hdr0 = hdulist[0].header
                    hdr1 = hdulist[1].header
                    data = hdulist[1].data

                    for i, segment in enumerate(data['segment']):
                        if segment not in data_dic[cenwave].keys():
                            data_dic[cenwave][segment] = {
                                'net': [],
                                'wavelength': [],
                                'grating': [],
                                'lp': [],
                                'target': [],
                                'rootname': [],
                                'date': [],
                                'infiles': []
                            }
                        
                        data_dic[cenwave][segment]['net'].append(np.array(data['net'][i][data['dq_wgt'][i] != 0]))
                        data_dic[cenwave][segment]['wavelength'].append(np.array(data['wavelength'][i][data['dq_wgt'][i] != 0]))

                        data_dic[cenwave][segment]['grating'].append(hdr0['opt_elem'])
                        data_dic[cenwave][segment]['lp'].append(hdr0['life_adj'])
                        data_dic[cenwave][segment]['target'].append(hdr0['targname'])
                        data_dic[cenwave][segment]['rootname'].append(hdr0['rootname'])
                        data_dic[cenwave][segment]['date'].append(Time(hdr1['date-obs'], format='fits').decimalyear)
                        data_dic[cenwave][segment]['infiles'].append(file)

        # change to np.array
        for cenwave in data_dic:
            for segment in data_dic[cenwave]:
                data_dic[cenwave][segment]['grating'] = np.array(data_dic[cenwave][segment]['grating'])
                data_dic[cenwave][segment]['lp'] = np.array(data_dic[cenwave][segment]['lp'])
                data_dic[cenwave][segment]['target'] = np.array(data_dic[cenwave][segment]['target'])
                data_dic[cenwave][segment]['rootname'] = np.array(data_dic[cenwave][segment]['rootname'])
                data_dic[cenwave][segment]['date'] = np.array(data_dic[cenwave][segment]['date'])
                data_dic[cenwave][segment]['infiles'] = np.array(data_dic[cenwave][segment]['infiles'])
        return (data_dic)

# --------------------------------------------------------------------------------#
    def parse_infiles(self, PIDs, csv_file, COSMO = '/grp/hst/cos2/cosmo/', pattern='*x1d.fits*'):
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
       # read in dat file that has the PIDs of all the FUV TDS monitoring data
       programs_df = pd.read_csv(PIDs, delim_whitespace=True)
       
       # change dat file into list of str of PID numbers
       all_programs = []
       for _, col_data in programs_df.items(): 
          all_programs += col_data.to_numpy(dtype=str).tolist()
       
       # get the paths to the x1d files from COSMO, multithreading
       with mp.Pool(16) as pool:
           x1d_paths = pool.starmap(self._get_x1ds, zip(repeat(COSMO), all_programs, repeat(pattern)))
       x1d_paths = [x for sublist in x1d_paths for x in sublist] # flatten list
       pool.terminate()

       # checks to see if an inventory csv file exists and what data is not
       # included that still needs to be ingested.
       # If no csv file exists, then a new inventory csv file will be created.
       if os.path.exists(csv_file):
           in_table = pd.read_csv(csv_file)
           x1d_paths = list(set(x1d_paths) - set(in_table))
       else:
           x1d_paths = list(x1d_paths)
    
       # get the data from the x1d files into several dataframe tables. Multithreading
       tables = process_map(self.get_x1ds_data, x1d_paths, max_workers=16, chunksize=10)
       
       # combine all the x1d dataframe tables into one
       tables = pd.concat(tables, ignore_index=True)

       # If dataframe tables were created, sort by date
       if len(tables) != 0:
           tables = tables.sort_values(by=['date-obs'], ignore_index=True)

           # if inventory csv file exists, add any new data into the 
           # pre-existing file.
           if os.path.exists(csv_file):
               tables.to_csv(csv_file, mode='w+')
           
           # if inventory csv file does not exist, create a new csv file
           # from scratch. User can see what datasets were used in the monitor
           # by this csv file.
           else:
               tables.to_csv(csv_file)
               print(f'{csv_file} was created.')
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
    
    def get_x1ds_data(self, file_path):
        """
        Open up the x1d files to obtain the necessary data to populate the csv file.

        Args:
            file_path: the path to the file we want to open up

        Returns:
            x1d_table: DataFrame of the information we want from the x1d file
        """

        # Applies to c1280_check. We do not use c1280 at fppos 3 in LP1. 
        # Only LP2 and onward. # MJD
        LP2_handoff = 56130.0

        # open file
        hdu = fits.open(file_path)
        exptime = hdu[1].header['exptime']

        # Use fppos 3 data only, except for c1280 if in LP1.
        # Do not use c1280 at fppos 3 in LP1, instead use fppos 4 for LP1.
        # After switch to LP2, use fppos 3. 
        def c1280_check(cenwave, fppos, expstart):
            good = False
            if (fppos == 3) & (cenwave != 1280):
                good = True
            elif (cenwave == 1280) & (fppos == 4) & (expstart < LP2_handoff):
                good = True
            elif (cenwave == 1280) & (fppos == 3) & (expstart > LP2_handoff):
                good = True
            return (good)
        
        def bad_list(grating):
            """
            These exposures have been manually removed from the FUV TDS dataset for a variety of reasons, including
            but not limited to: bad aperture centering, missing wavelength arrays, no longer used cenwaves, etc.
            As COSMO stores all datasets, this bad_roots dictionary filters out exposures no longer used in the FUV TDS
            dataset. 

            At some point, someone can find similarities between these datasets to include instead in the criteria
            dictionary instead of explicitly writing not to use.

            In the meantime, this 'bad_roots' list will have to be manually updated if/when more exposures fail that
            escape the HOPPER alert.
            """
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
            return (bad_roots[grating])
        
        # dictionary of if statements to filter out the data for exposures we do not use.
        criteria = {
            'exptime': exptime != 0,
            'bad_targs': hdu[0].header['targname'] not in ['WAVE', 'LDS749B'],
            'bad_items': hdu[0].header['rootname'] not in bad_list(hdu[0].header['opt_elem']),
            'fppos_check': c1280_check(
                hdu[0].header['cenwave'],
                hdu[0].header['fppos'],
                hdu[1].header['expstart']) == True,
            'wl': len(hdu[1].data['wavelength']) != 0,
            'lp4': (hdu[0].header['opt_elem'] != 'G160M') | ((hdu[0].header['life_adj'] != 4) | (hdu[1].header['date-obs'] < '2022-10-01')),
            'cenwave': (hdu[0].header['cenwave'] in self.cenwaves) | (hdu[0].header['cenwave'] == 1230)
        }

        # if-statement to filter out exposures we do not use and place the datasets we DO use
        #into a dataframe to then be turned into a csv file data product.
        if (criteria['exptime']) & (criteria['bad_targs']) & (criteria['bad_items']) & (criteria['fppos_check']) & (criteria['wl']) & (criteria['lp4']) & (criteria['cenwave']):
            if hdu[0].header['cenwave'] == 1230:
                cenwave = 1280
            else: cenwave = hdu[0].header['cenwave']
            x1d_table = pd.DataFrame(
                {
                    'rootname': [hdu[0].header['rootname']],
                    'opt_elem': [hdu[0].header['opt_elem']],
                    'cenwave': [cenwave],
                    'segment': [hdu[0].header['segment']],
                    'fppos': [hdu[0].header['fppos']],
                    'life_adj': [hdu[0].header['life_adj']],
                    'proposid': [hdu[0].header['proposid']],
                    'targname': [hdu[0].header['targname']],
                    'date-obs': [hdu[1].header['date-obs']],
                    'exptime': [exptime],
                    'file_path': [file_path]
                }
            )
            hdu.close()
            return (x1d_table)