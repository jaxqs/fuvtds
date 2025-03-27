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
import mpfit
from scipy.interpolate import interp1d

import plotly.graph_objs as go

import requests
from io import StringIO

"""
This is the base class for the FUVTDS Monitor that will do all
the analysis need and provide the necessary component to the
self that will conduct all the plotting.

Please be aware, this will be a monster.

Notes to do in the future:
    - Exchange mpfit.mpfit with scipy.curve_fit
    - Put outputs into a log file instead out outright outputting
    - csv file of Bad exposures instead of a dictionary
    - add in latest pids and exposures and bad exposures
    - add latest hv raise
    - code in the target change
"""

__author__ = 'J. Hernandez' #Me! JAQ!

class FUVTDSBase:
    """
    This class will analyze and store the results necessary to
    conduct a FUVTDS self analysis.

    Attributes:
        PIDs: .dat file of TDS PIDs
        reftime: the 'zero' point in time, 2003
        inventory: csv file that will save all data used in fuv tds analysis
    """

    def __init__(self, PIDs, reftime = 54952.0, inventory='inventory.csv'):
        """
        Args:
        """
        self.breakpoints = np.array([2010.2, 2011.2, 2011.75, 2012.0, 2012.8, 2013.8, 2015.5, 2019.0, 2020.6, 2022.0, 2023.2])
        self.HV_FUVA = np.array([2012.23,2012.56,2014.84,2015.107,2017.75,2020.75,2021.76, 2023.94])
        self.HV_FUVB = np.array([2011.18,2013.47,2012.56,2014.55,2015.107,2016.05,2017.75,2020.75,2022.47, 2023.94])
        self.LPs = np.array([2012.56, 2015.107, 2017.75, 2021.76, 2022.75])
        self.cenwaves = np.array([1533, 1577, 1623, 1291, 1327, 1105, 1280, 800, 1222, 1055, 1096])
        self.segments = ['FUVA', 'FUVB']
        self.reftime = Time(reftime, format="mjd").decimalyear

        tables = self.parse_infiles(PIDs, inventory)
        self.tables = self.scalings(tables)
        self.TDSDates = self.important_dates()
    
    def important_dates(self):
        TDSDates = {
            'breakpoints': self.breakpoints,
            'HV_FUVA': self.HV_FUVA,
            'HV_FUVB': self.HV_FUVB,
            'LPs': self.LPs,
            'reftime': self.reftime
        }

        return(TDSDates)

# --------------------------------------------------------------------------------#
    def scalings(self, table):
        """
        """
        tables = []
        for segment in set(table['segment']):
            with mp.Pool(16) as pool:
                tab = pool.starmap(self.scale_lps, zip(set(table['cenwave']), repeat(segment), repeat(table)))
            pool.terminate

            tab = pd.concat(tab, ignore_index=True)
            tables.append(tab)
            
        new_table = pd.concat(tables, ignore_index=True)
        new_table = new_table.sort_values(by=['date-obs'], ignore_index=True)

        return(new_table)
        
# --------------------------------------------------------------------------------#
    def scale_lps(self, cenwave, segment, table):
        """
        """

        def find_nearest(array, value):
            array = np.asarray(array)
            idx = (np.abs(array - value)).argmin()
            return (idx)
        
        def scale(scaled, scaled_to, a):
            size = ['small', 'large']

            for size in size:
                for i, _ in enumerate(scaled[f'{size}_binned_wl'].iloc[0]):

                    scaled_error = scaled[f'{size}_stdev'].iloc[0][i]
                    scaled_data  = scaled[f'{size}_binned_net'].iloc[0][i] 

                    scaled_to_error = scaled_to[f'{size}_stdev'].iloc[a][i]
                    scaled_to_data  = scaled_to[f'{size}_binned_net'].iloc[a][i]
                    
                    scale_factor = (scaled_to_data/
                                    scaled_data)
                    
                    scale_factor_stdev = np.sqrt(
                        (scaled_to_error/scaled_data)**2 +
                        (scaled_to_data/(scaled_data**2)*scaled_error)**2
                    )
                    
                    for j, _ in enumerate(scaled[f'{size}_binned_net']):

                        scaled[f'{size}_scaled_stdev'].iloc[j][i] = np.sqrt(
                            scale_factor**2 *
                            scaled[f'{size}_stdev'].iloc[j][i]**2 +
                            scaled[f'{size}_binned_net'].iloc[j][i]**2 *
                            scale_factor_stdev**2
                        )

                        scaled[f'{size}_scaled_net'].iloc[j][i] = (
                            scaled[f'{size}_binned_net'].iloc[j][i]*
                            scale_factor
                        )
                
            return(scaled)

        table = table[(table['cenwave'] == cenwave) & (table['segment'] == segment)]


        if len(table) == 0:
            return
        
        # Scale the change in target for G160M LP6 and G130M/1096/FUVB LP2
        if ( (6 in np.array(table['life_adj']).flatten()) | (2 in np.array(table['life_adj']).flatten()) ) & ('WD1057+719' in np.array(table['targname']).flatten()):
            """
            Fit the LP6 G160M GD71 data and the WD1057+719 data separately and use the difference
            at the first WD1057+719 date to scale all the WD1057+719 data to the LP6 G160M GD71 data
            """

            lp6_wd1057 = table[(table['life_adj'] == 6) & (table['targname'] == 'WD1057+719')]
            
            lp6_gd71 = table[(table['life_adj'] == 6) & (table['targname'] == 'GD71')]
            
            lp6_wd308 = table[(table['life_adj'] == 6) & (table['targname'] == 'WD0308-565')]
            
            if (cenwave > 1500) & (segment == 'FUVA'):
                old_target = lp6_gd71
                new_target = lp6_wd1057
            elif (cenwave == 1096) & (segment == 'FUVB'):
                old_target = table[(table['life_adj'] == 2) & (table['targname'] == 'GD71')]
                new_target = table[(table['life_adj'] == 2) & (table['targname'] == 'WD1057+719')]
            else:
                old_target = lp6_wd308
                new_target = lp6_wd1057
            
            a = find_nearest(np.array(old_target['date-obs']).flatten(), 
                             np.array(new_target['date-obs']).flatten()[0])
            
            new_target = scale(new_target, old_target, a)

            if (6 in np.array(table['life_adj']).flatten()) & ('WD1057+719' in np.array(table['targname']).flatten()):
                table.loc[(table['life_adj'] == 6) & (table['targname'] == 'WD1057+719' )] = new_target
            elif (2 in np.array(table['life_adj']).flatten()) & ('WD1057+719' in np.array(table['targname']).flatten()):
                table.loc[(table['life_adj'] == 2) & (table['targname'] == 'WD1057+719' )] = new_target
            
            print(f"+++ Scaling old target to WD1057+719 using data from datasets: {old_target['file_path'].iloc[a]} {new_target['file_path'].iloc[0]}")
            
        
        if (6 in np.array(table['life_adj']).flatten()) & (4 in np.array(table['life_adj']).flatten()):
            
            lp6_wd308 = table[(table['life_adj'] == 6) &
                              (table['targname'] == 'WD0308-565')]
            lp6_gd71 = table[(table['life_adj'] == 6) &
                                  (table['targname'] == 'GD71')]
            
            lp4_wd308 = table[(table['life_adj'] == 4) &
                              (table['targname'] == 'WD0308-565')]
            lp4_gd71 = table[(table['life_adj'] == 4) &
                             (table['targname'] == 'GD71')]
            
            if (cenwave > 1500) & (segment == 'FUVA'):
                lp4 = lp4_gd71
                lp6 = lp6_gd71
            else: 
                lp4 = lp4_wd308
                lp6 = lp6_wd308

            #Find the connection visit by closest date. Only a safe assumption for LP5 and 
            #LP6 where the visits happened on the same day.
            a = find_nearest(np.array(lp4['date-obs']).flatten(), 
                            np.array(lp6['date-obs']).flatten()[0])
        
            lp6 = scale(lp6, lp4, a)

            table.loc[table['life_adj'] == 6] = lp6
            

            print(f"+++ Scaling LP6 to LP4 using data from datasets: {lp4['file_path'].iloc[a]} {lp6['file_path'].iloc[0]}")

        if (5 in np.array(table['life_adj']).flatten()) & (4 in np.array(table['life_adj']).flatten()):

            lp5_wd308 = table[(table['life_adj'] == 5) &
                                    (table['targname'] == 'WD0308-565')]
            lp5_gd71 = table[(table['life_adj'] == 5) &
                                    (table['targname'] == 'GD71')]
            
            lp4_wd308 = table[(table['life_adj'] == 4) &
                                    (table['targname'] == 'WD0308-565')]
            lp4_gd71 = table[(table['life_adj'] == 4) &
                                    (table['targname'] == 'GD71')]
            
            if (cenwave > 1500) & (segment == 'FUVA'):
                lp4 = lp4_gd71
                lp5 = lp5_gd71
            else: 
                lp4 = lp4_wd308
                lp5 = lp5_wd308

            a = find_nearest(np.array(lp4['date-obs']).flatten(), 
                            np.array(lp5['date-obs']).flatten()[0])
            
            lp5 = scale(lp5, lp4, a)
            table.loc[table['life_adj'] == 5] = lp5

            print(f"+++ Scaling LP5 to LP4 using data from datasets: {lp4['file_path'].iloc[a]} {lp5['file_path'].iloc[0]}")
        

        # LP3 to LP4 to LP3
        if (4 in np.array(table['life_adj']).flatten()) & (3 in np.array(table['life_adj']).flatten()):
            
            # scale lp3 to lp4
            lp4_wd308 = table[(table['life_adj'] >= 4) &
                            (table['targname'] == 'WD0308-565')]
            lp4_gd71 = table[(table['life_adj'] >= 4) &
                            (table['targname'] == 'GD71')]
            
            lp3_wd308 = table[(table['life_adj'] == 3) &
                                (table['targname'] == 'WD0308-565') &
                                (table['date-obs'] < 2019.0)]
            lp3_gd71 = table[(table['life_adj'] == 3) &
                                (table['targname'] == 'GD71')&
                                (table['date-obs'] < 2019.0)]
            
            lp3_wd308_2 = table[(table['life_adj'] == 3) &
                                (table['targname'] == 'WD0308-565') &
                                (table['date-obs'] > 2019.0)]
            lp3_gd71_2 = table[(table['life_adj'] == 3) &
                                (table['targname'] == 'GD71')&
                                (table['date-obs'] > 2019.0)]
            
            if (cenwave > 1500) & (segment == 'FUVA'):
                lp3   = lp3_gd71 #LP3 indicies after LP3->LP4->LP3.
                lp4   = lp4_gd71
                lp3_2 = lp3_gd71_2
            else: 
                lp3   = lp3_wd308 #LP3 indicies after LP3->LP4->LP3.
                lp4   = lp4_wd308
                lp3_2 = lp3_wd308_2
            
            if len(lp3_2) != 0:
                lp4 = pd.concat([lp4, lp3_2]) #lp4_indx needs to contain the post 2021 lp3_indx
            
            if cenwave != 800:
                a = find_nearest(np.array(lp3['date-obs']).flatten(), 
                            np.array(lp4['date-obs']).flatten()[0])
            
                lp4 = scale(lp4, lp3, a)

                table.loc[(table['life_adj'] >= 4) | ((table['life_adj'] == 3) & (table['date-obs'] > 2019.0))] = lp4

                print(f"+++ Scaling LP4 to LP3 using data from datasets: {lp3['file_path'].iloc[a]} {lp4['file_path'].iloc[0]}")
            
                if len(lp3_2) != 0:
                    size = ['small', 'large']

                    for size in size:
                        for i, _ in enumerate(lp3_2[f'{size}_binned_wl'].iloc[0]):

                            scaled_error = lp3[f'{size}_stdev'].iloc[a][i]
                            scaled_data  = lp3[f'{size}_binned_net'].iloc[a][i]

                            scaled_to_error = lp4[f'{size}_stdev'].iloc[0][i]
                            scaled_to_data  = lp4[f'{size}_binned_net'].iloc[0][i]
                            
                            scale_factor = (scaled_to_data/
                                            scaled_data)
                            scale_factor_stdev = np.sqrt(
                                (scaled_to_error/scaled_data)**2 +
                                (scaled_to_data/(scaled_data**2)*scaled_error)**2
                            )
                            
                            for j, _ in enumerate(lp3_2[f'{size}_binned_net']):

                                lp3_2[f'{size}_scaled_stdev'].iloc[j][i] = np.sqrt(
                                    scale_factor**2 *
                                    lp3_2[f'{size}_stdev'].iloc[j][i]**2 +
                                    lp3_2[f'{size}_binned_net'].iloc[j][i]**2 *
                                    scale_factor_stdev**2
                                )
        
                                lp3_2[f'{size}_scaled_net'].iloc[j][i] = (
                                    lp3_2[f'{size}_binned_net'].iloc[j][i]*
                                    scale_factor)

                    table.loc[(table['life_adj'] == 3) & (table['date-obs'] > 2019.0)] = lp3_2
                    print(f"+++ Scaling LP3 to LP4 using data from datasets: {lp4['file_path'].iloc[0]} {lp3['file_path'].iloc[a]}")
                
            elif cenwave == 800:

                a = find_nearest(np.array(lp4['date-obs']).flatten(),
                                 np.array(lp3_2['date-obs']).flatten()[0])
                size = ['small', 'large']

                for size in size:
                    for i, _ in enumerate(lp3_2[f'{size}_binned_wl'].iloc[0]):
                        scaled_error = lp3_2[f'{size}_stdev'].iloc[0][i]
                        scaled_data  = lp3_2[f'{size}_binned_net'].iloc[0][i] 

                        scaled_to_error = lp4[f'{size}_stdev'].iloc[a][i]
                        scaled_to_data  = lp4[f'{size}_binned_net'].iloc[a][i]
                        
                        scale_factor = (scaled_to_data/
                                        scaled_data)
                        scale_factor_stdev = np.sqrt(
                            (scaled_to_error/scaled_data)**2 +
                            (scaled_to_data/(scaled_data**2)*scaled_error)**2
                        )
                    
                        for j, _ in enumerate(lp3_2[f'{size}_binned_net']):
                            lp3_2[f'{size}_scaled_stdev'].iloc[j][i] = np.sqrt(
                                scale_factor**2 *
                                lp3_2[f'{size}_stdev'].iloc[j][i]**2 +
                                lp3_2[f'{size}_binned_net'].iloc[j][i]**2 *
                                scale_factor_stdev**2
                                )

                            lp3_2[f'{size}_scaled_net'].iloc[j][i] = (
                                lp3_2[f'{size}_binned_net'].iloc[j][i]*
                                scale_factor)

                table.loc[table['life_adj'] == 3] = lp3_2

                print(f"+++ Scaling LP3 to LP4 using data from datasets: {lp4['file_path'].iloc[a]} {lp3_2['file_path'].iloc[0]}")
        


        if (3 in np.array(table['life_adj']).flatten()) & (2 in np.array(table['life_adj']).flatten()):

            lp3_wd308 = table[(table['life_adj'] >= 3) &
                                    (table['targname'] == 'WD0308-565')]
            lp3_gd71 = table[(table['life_adj'] >= 3) &
                                    (table['targname'] == 'GD71')]
            
            lp2_wd308 = table[(table['life_adj'] == 2) &
                                    (table['targname'] == 'WD0308-565')]
            lp2_gd71 = table[(table['life_adj'] == 2) &
                                    (table['targname'] == 'GD71')]
            
            if (cenwave > 1500) & (segment == 'FUVA'):
                lp2 = lp2_gd71
                lp3  = lp3_gd71
            else: 
                lp2 = lp2_wd308
                lp3  = lp3_wd308

            a = find_nearest(np.array(lp2['date-obs']).flatten(), 
                            np.array(lp3['date-obs']).flatten()[0])
            
            lp3 = scale(lp3, lp2, a)
            table.loc[table['life_adj'] >= 3] = lp3

            print(f"+++ Scaling LP3 to LP2 using data from datasets: {lp2['file_path'].iloc[a]} {lp3['file_path'].iloc[0]}")
        

        if (2 in np.array(table['life_adj']).flatten()) & (1 in np.array(table['life_adj']).flatten()):
            lp2_wd308 = table[(table['life_adj'] >= 2) &
                                   (table['targname'] == 'WD0308-565')]
            lp2_gd71 = table[(table['life_adj'] >= 2) &
                                  (table['targname'] == 'GD71')]
            
            lp1_wd1057 = table[(table['life_adj'] == 1) &
                                    (table['targname'] == 'WD1057+719')]
            lp1_wd0947 = table[(table['life_adj'] == 1) &
                                    (table['targname'] == 'WD0947+857')]
            
            
            if (cenwave > 1500) & (segment == 'FUVA'):
                lp1 = lp1_wd1057
                lp2 = lp2_gd71
            elif (cenwave > 1500) & (segment == 'FUVB'):
                lp1 = lp1_wd1057
                lp2 = lp2_wd308
            else: 
                lp1 = lp1_wd0947
                lp2  = lp2_wd308

            a = find_nearest(np.array(lp1['date-obs']).flatten(), 
                            np.array(lp2['date-obs']).flatten()[0])
            
            lp2 = scale(lp2, lp1, a)

            table.loc[table['life_adj'] >= 2] = lp2

            print(f"+++ Scaling LP2 to LP1 using data from datasets: {lp1['file_path'].iloc[a]} {lp2['file_path'].iloc[0]}")
        return (table)

# --------------------------------------------------------------------------------#
    def hdu_bin(self, file):
        """
        """

        targ_info_dict = {
            1533: {'FUVA': ['GD71', 'WD1057+719'], 'FUVB': ['WD1057+719', 'WD0308-565']},
            1577: {'FUVA': ['GD71', 'WD1057+719'], 'FUVB': ['WD1057+719', 'WD0308-565']},
            1623: {'FUVA': ['GD71', 'WD1057+719'], 'FUVB': ['WD1057+719', 'WD0308-565']},
            1291: {'FUVA': ['WD0308-565', 'WD0947+857'], 'FUVB': ['WD0308-565', 'WD0947+857']},
            1327: {'FUVA': ['WD0308-565', 'WD0947+857'], 'FUVB': ['WD0308-565', 'WD0947+857']},
            1105: {'FUVA': ['WD0308-565', 'WD0947+857']},
            1280: {'FUVA': ['WD0308-565', 'WD0947+857'], 'FUVB': ['WD0308-565', 'WD0947+857']},
            800:  {'FUVA': ['WD0308-565']},
            1222: {'FUVA': ['WD0308-565'], 'FUVB': ['WD0308-565']},
            1055: {'FUVA': ['WD0308-565'], 'FUVB': ['WD0308-565']},
            1096: {'FUVB': ['GD71', 'WD1057+719']}
        }

        # Dictionary that sets the binsize and wavelength edges of each segment
        # as the wavelength edges changes depending if the size is small or large.
        # Values based on the wl_info_dict dictionary in original FUV TDS Monitor.
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

        # Applies to c1280_check. We do not use c1280 at fppos 3 in LP1. 
        # Only LP2 and onward. # MJD
        LP2_handoff = 56130.0

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
                           'lbxmt3m1q', 'lf4g8aj5q', 'lf4g06fhq', 'lf4h06xjq', 'lf4h5agqq', 'lf4g06fhq'], 
                'G140L': ['ldv007piq', 'ldv008onq', 'ldv010fpq', 'le5g07n4q', 'ler158owq', 'lf205ag2q',
                          'ldqj05ymq', 'ldqj08jbq', 'ldqj12eeq', 'ldqj56afq', 'ldqj57u3q', 'ldqj58hpq',
                          'ldqj59k5q', 'ldv007pkq', 'ldv008opq', 'ldv010ftq', 'le5g07n8q', 'lf205ag4q',
                          'ldqj05ykq', 'ldqj08j9q', 'ldqj12ecq', 'ldqj56adq', 'ldqj57u1q', 'ldqj58hmq',
                          'ldqj59k3q', 'ldv007paq', 'ldv010evq', 'lefe03grq', 'lbxmt1nzq', 'lbxmt3lwq',
                          'lbxmt3mfq', 'lf4h8ar8q', 'lf4h8araq', 'lf4h5agiq', 'lf4h5agoq', 'lf4h5agmq'],
                'G160M': ['lbb917kfq', 'ldv006lqq', 'ldv007pcq', 'ler106dqq', 'lf2056ftq', 'ldqj05y0q', 
                          'ldqj08j3q', 'ldqj13e9q', 'ldqj12e8q', 'ldqj56a9q', 'ldqj57txq', 'ldqj58hgq', 
                          'ldv006lsq', 'ldqj59jzq', 'ldv007peq', 'ldv008ojq', 'ldv010flq', 'ler106dsq',
                          'lf2006liq', 'lf205bauq', 'lf2056g5q', 'ldqj05y2q', 'ldqj08j5q', 'ldqj13ebq', 
                          'ldqj12eaq', 'ldqj56abq', 'ldqj57tzq', 'ldqj58hjq', 'ldv006lvq', 'ldqj59k1q', 
                          'ldv007pgq', 'ldv008olq', 'ldv010fnq', 'ler106doq', 'ler106dwq', 'lf2006lqq', 
                          'lf205bb8q', 'lf2111zuq', 'ler106duq', 'ler106dkq', 'lbxm02agq', 'lbxm02bxq', 
                          'lf2111zoq', 'lf2006lmq', 'lbxm02ayq', 'lf205bb1q', 'lf4g06fnq', 'lf4h06y5q',
                          'lf4h06y1q', 'lf4h06ydq', 'lf4h5bn8q', 'lf4h5bn2q', 'lf4h5bnkq', 'lf4g06fnq', 
                          'lf4g07cqq']} 
            return (bad_roots[grating])
   

        tables = []

        with fits.open(file, memmap=False) as hdulist:
            hdr0 = hdulist[0].header
            hdr1 = hdulist[1].header
            data = hdulist[1].data

            # dictionary of if statements to filter out the data for exposures we do not use.
            criteria = {
                'exptime': hdr1['exptime'] != 0,
                'bad_targs': hdr0['targname'] not in ['WAVE', 'LDS749B'],
                'bad_items': hdr0['rootname'] not in bad_list(hdr0['opt_elem']),
                'fppos_check': c1280_check(
                    hdr0['cenwave'],
                    hdr0['fppos'],
                    hdr1['expstart']) == True,
                'wl': len(data['wavelength']) != 0,
                'lp4': (hdr0['opt_elem'] != 'G160M') | ((hdr0['life_adj'] != 4) | (hdr1['date-obs'] < '2022-10-01')),
                'cenwave': (hdr0['cenwave'] in targ_info_dict.keys()) | (hdr0['cenwave'] == 1230)
            }

            if (criteria['exptime']) & (criteria['bad_targs']) & (criteria['bad_items']) & (criteria['fppos_check']) & (criteria['wl']) & (criteria['lp4']) & (criteria['cenwave']):
                if hdr0['cenwave'] == 1230:
                    cenwave = 1280
                else: cenwave = hdr0['cenwave']

                # Iterate over the segements of the x1d file. If theres only one segment
                # used, this will only iterate once. If two segments are used, then this
                # will iterate twice. 
                # data['segment'] will list all segments used. This also applies to NUV
                # data as well.
                for i, segment in enumerate(data['segment']):   
                    
                    if hdr0['targname'] not in targ_info_dict[cenwave][segment]:
                        continue
                    # wl_range contains minimum wavelength, maximum wavelength, and binsize
                    small_min_wl, small_max_wl, small_binsize = wl_info_dict['small'][cenwave][segment]

                    # the wavelength bin edges based on binsize
                    small_bins = np.arange(small_min_wl, small_max_wl+1, small_binsize)

                    # wl_range contains minimum wavelength, maximum wavelength, and binsize
                    large_min_wl, large_max_wl, large_binsize = wl_info_dict['large'][cenwave][segment]

                    # the wavelength bin edges based on binsize
                    large_bins = np.arange(large_min_wl, large_max_wl+1, large_binsize)

                    dqwgt = data['dq_wgt'][i] != 0
                    wl    = data['wavelength'][i][dqwgt]
                    net   = data['net'][i][dqwgt]

                    small_x_index = np.where((wl >= small_min_wl) & (wl < small_max_wl))
                    large_x_index = np.where((wl >= large_min_wl) & (wl < large_max_wl))

                    # determine the mean net count rate for each bin
                    small_mean_net, small_edges, _ = binned_statistic(
                        wl[small_x_index],
                        net[small_x_index],
                        "mean", bins=small_bins
                    )
                    large_mean_net, large_edges, _ = binned_statistic(
                        wl[large_x_index],
                        net[large_x_index],
                        "mean", bins=large_bins
                    )

                    # determine the std for each bin
                    small_stdev = binned_statistic(
                        wl[small_x_index],
                        net[small_x_index],
                        "std", bins=small_bins
                    )[0] / np.sqrt(len(net[small_x_index]))

                    large_stdev = binned_statistic(
                        wl[large_x_index],
                        net[large_x_index],
                        "std", bins=large_bins
                    )[0] / np.sqrt(len(net[large_x_index]))

                    x1d_table = pd.DataFrame(
                        {
                            'rootname': [hdr0['rootname']],
                            'opt_elem': [hdr0['opt_elem']],
                            'cenwave': [cenwave],
                            'segment': [segment],
                            'fppos': [hdr0['fppos']],
                            'life_adj': [hdr0['life_adj']],
                            'proposid': [hdr0['proposid']],
                            'targname': [hdr0['targname']],
                            'date-obs': [Time(hdr1['date-obs'], format='fits').decimalyear],
                            'date-obs-fits': [hdr1['date-obs']],
                            'exptime': [hdr1['exptime']],
                            'file_path': [file],
                            'small_binned_net': [small_mean_net],
                            'small_scaled_net': [small_mean_net],
                            'small_stdev': [small_stdev],
                            'small_scaled_stdev': [small_stdev],
                            'small_binned_wl': [small_edges[:-1]+np.diff(small_edges)/2],
                            'small_wl_edges': [small_edges],
                            'large_binned_net': [large_mean_net],
                            'large_scaled_net': [large_mean_net],
                            'large_stdev': [large_stdev],
                            'large_scaled_stdev': [large_stdev],
                            'large_binned_wl': [large_edges[:-1]+np.diff(large_edges)/2],
                            'large_wl_edges': [large_edges]
                            }
                        )
                    tables.append(x1d_table)
                if len(tables) != 0:
                    table = pd.concat(tables)
                    return table

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
       # read in dat file that has the PIDs of all the FUV TDS selfing data
       programs_df = pd.read_csv(PIDs)
       
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
    
       # get data from x1d files and bin
       tables = process_map(self.hdu_bin, x1d_paths, max_workers=16, chunksize=10)
       
       # combine all the x1d dataframe tables into one
       tables = pd.concat(tables, ignore_index=True)

       # If dataframe tables were created, sort by date
       if len(tables) != 0:
           tables = tables.sort_values(by=['date-obs-fits'], ignore_index=True)
           
           csv_file_save = tables.drop(columns=['small_binned_net',
                                                'small_scaled_net',
                                                'small_binned_wl', 
                                                'small_wl_edges',
                                                'large_binned_net', 
                                                'large_scaled_net',
                                                'large_binned_wl', 
                                                'large_wl_edges',
                                                'small_stdev',
                                                'small_scaled_stdev',
                                                'large_stdev',
                                                'large_scaled_stdev',
                                                'date-obs'])
           csv_file_save = csv_file_save.rename(columns={'date-obs-fits': 'date-obs'})
           #tables = tables.drop(columns=['date-obs-fits'])
           if os.path.exists(csv_file):
               csv_file_save.to_csv(csv_file, mode='w+')
           else:
               csv_file_save.to_csv(csv_file)
               print(f'{csv_file} was created.')
        
           return(tables)
# --------------------------------------------------------------------------------#
    def _get_x1ds(self, COSMO, all_programs, pattern):
        """
        Obtain the full path to the x1d files from COSMO directory.
        Used solely for the parse_infiles(function).

        Args:
            COSMO: the directory in which the data is stored. By default,
                    this will be /grp/hst/cos2/cosmo.
            all_programs: the PIDs of all the programs used in the FUVTDS self
            pattern: the pattern of file we want, in this case x1d files.
        Returns:
            path_list: the path to the x1d file.
        """
        total_path = os.path.join(COSMO, all_programs, pattern)
        path_list = glob.glob(total_path)
        return(path_list)


class FUVTDSMonitor(object):

    def __init__(self, TDSDates):
        # the vertical lines used in plots
        self.breakpoints = np.array(TDSDates['breakpoints'])
        self.HV_FUVA = np.array(TDSDates['HV_FUVA'])
        self.HV_FUVB = np.array(TDSDates['HV_FUVB'])
        self.LPs = np.array(TDSDates['LPs'])
        self.reftime = TDSDates['reftime']
# --------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# PLOT TOOLS
# --------------------------------------------------------------------------------#
    def rel_sens_graph(self, date, net, net_err, i, wl_edges, best_fit_model, df, tds, fig):
        """
        This tool is so I can reduce repetition in the app. This way one function will be called
        to display the relative sensitivity graph in the dash app and to save as multiple html
        files. This will reduce redundance of editing the same plotly code twice in different
        places. 
        """

        segment = df['segment'].iloc[0]

        curr_sens = go.Scatter(
            x = date,
            y = net[:,i],
            error_y = dict(type='data', array=net_err[:,i], visible=True),
            mode='markers',
            name=f'Current TDS {wl_edges[i]} - {wl_edges[i+1]}',
            customdata= np.stack(
                (np.array(df['rootname']).flatten(),
                np.array(df['life_adj']).flatten(),
                np.array(df['proposid']).flatten(),
                np.array(df['targname']).flatten(),
                np.array(df['date-obs-fits']).flatten()),
                axis=-1
            ),
            hovertemplate=
            'Rootname: %{customdata[0]}<br>'+
            'Life_adj: %{customdata[1]}<br>'+
            'Proposid: %{customdata[2]}<br>'+
            'Target: %{customdata[3]}<br>'+
            'Date Obs: %{customdata[4]}'
            "<extra></extra>",
            line_color='black'
        )

        # tds model
        tds_model = go.Scatter(
            x = date,
            y = tds[i],
            name = 'Current TDSTAB',
            line = dict(color='orange', width=4, dash='dash')
        )

        x = np.append(self.reftime, self.breakpoints)
        x = np.append(x, date[-1])

        # best fit 
        best_fit = go.Scatter(
            x = x, 
            y = self.broken_lines(x - self.reftime, *best_fit_model[i,:]),
            name = 'Best Fit',
            line = dict(color='grey', width=4, dash='dash')
        )

        # plot two! plot two!
        ymodel = tds[i]
        residual = 100.0*(net[:,i] - ymodel) / ymodel
        sens_err = go.Scatter(
            x = date, 
            y = residual,
            mode = 'markers',
            name = 'TDSTAB',
            customdata = np.stack(
                (np.array(df['rootname']).flatten(),
                np.array(df['life_adj']).flatten(),
                np.array(df['proposid']).flatten(),
                np.array(df['targname']).flatten(),
                np.array(df['date-obs-fits']).flatten()),
                axis=-1
            ),
            hovertemplate=
            'Rootname: %{customdata[0]}<br>'+
            'Life_adj: %{customdata[1]}<br>'+
            'Proposid: %{customdata[2]}<br>'+
            'Target: %{customdata[3]}<br>'+
            'Date Obs: %{customdata[4]}'
            "<extra></extra>",
            line_color='orange'
        )

        ymodel = self.broken_lines(date - self.reftime, *best_fit_model[i,:])
        residual = 100.0 * (net[:,i] - ymodel) / ymodel
        best_fit_err = go.Scatter(
            x = date,
            y = residual,
            mode = 'markers',
            name = 'Best Fit',
            customdata = np.stack(
                (np.array(df['rootname']).flatten(),
                np.array(df['life_adj']).flatten(),
                np.array(df['proposid']).flatten(),
                np.array(df['targname']).flatten(),
                np.array(df['date-obs-fits']).flatten()),
                axis=-1
            ),
            hovertemplate =
            'Rootname: %{customdata[0]}<br>'+
            'Life_adj: %{customdata[1]}<br>'+
            'Proposid: %{customdata[2]}<br>'+
            'Target: %{customdata[3]}<br>'+
            'Date Obs: %{customdata[4]}'
            "<extra></extra>",
            line_color='black'
        )


        # fill between
        fig.add_trace(go.Scatter(
            x = [x[0]-0.1, x[-1]+0.1],
            y = [-5, -5],
            mode='lines',
            fill = 'tonexty',
            showlegend=False,
            line_color='teal'), row=2, col=1)
        fig.add_trace(go.Scatter(
            x = [x[0]-0.1, x[-1]+0.1],
            y = [5, 5],
            mode='lines',
            fill = 'tonexty',
            showlegend=False,
            line_color='teal'), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x = [x[0]-0.1, x[-1]+0.1],
            y = [-2, -2],
            mode='lines',
            fill = 'tozeroy',
            showlegend=False,
            line_color='purple'), row=2, col=1)
        fig.add_trace(go.Scatter(
            x = [x[0]-0.1, x[-1]+0.1],
            y = [2, 2],
            mode='lines',
            fill = 'tozeroy',
            showlegend=False,
            line_color='purple'), row=2, col=1)
        

        # add traces
        fig.add_trace(curr_sens, row=1, col=1)
        fig.add_trace(best_fit, row=1, col=1)
        fig.add_trace(tds_model, row=1, col=1)


        # add vertical lines here
        fig.add_traces(self.add_lines(self.breakpoints, dict(color='red', width=2, dash='dash'), 'Breakpoint', [min(net[:,i])-0.2, max(net[:,i])+0.2]))
        fig.add_traces(self.add_lines(self.breakpoints, dict(color='red', width=2, dash='dash'), 'Breakpoint', [min(residual)-3, max(residual)+3],True), rows=2, cols=1)
        fig.add_traces(self.add_lines(self.LPs, dict(color='grey', width=2, dash='dot'), 'LP switch', [min(net[:,i])-0.2, max(net[:,i])+0.2]))
        fig.add_traces(self.add_lines(self.LPs, dict(color='grey', width=2, dash='dot'), 'LP switch', [min(residual)-3, max(residual)+3], True), rows=2, cols=1)

        if segment == 'FUVA':
            fig.add_traces(self.add_lines(self.HV_FUVA, dict(color='purple', width=2, dash='dash'), 'Voltage Change SegA', [min(net[:,i])-0.2, max(net[:,i])+0.2]))
            fig.add_traces(self.add_lines(self.HV_FUVA, dict(color='purple', width=2, dash='dash'), 'Voltage Change SegA',[min(residual)-3, max(residual)+3], True), rows=2, cols=1)
        else:
            fig.add_traces(self.add_lines(self.HV_FUVB, dict(color='grey', width=2, dash='dash'), 'Voltage Change SegB', [min(net[:,i])-0.2, max(net[:,i])+0.2]))
            fig.add_traces(self.add_lines(self.HV_FUVB, dict(color='grey', width=2, dash='dash'), 'Voltage Change SegB', [min(residual)-3, max(residual)+3], True), rows=2, cols=1)
        
        # add more traces
        fig.add_trace(sens_err, row=2, col=1)
        fig.add_trace(best_fit_err, row=2, col=1)

        # update the y-axis based on what mode is being plotted
        fig.update_yaxes(range=(min(net[:,i])-0.2, max(net[:,i])+0.2), row=1, col=1),
        fig.update_yaxes(range=(min(residual)-3, max(residual)+3), row=2, col=1)
        
        return fig, x
    

    def time_slope_graph(self, df, size, timebin, fig):

        # Differentiate symbols - might be a better way to do this?
        marker_type = {'G140L': {'FUVA': 'square-open', 'FUVB': 'square'},
                    'G130M': {'FUVA': 'circle-open', 'FUVB': 'circle'},
                    'G160M': {'FUVA': 'triangle-up-open', 'FUVB': 'triangle-up'}}
        color_type = {'G140L': 'red',
                    'G130M': 'blue',
                    'G160M': 'teal'}
        counter_mode = {'G140L': {'FUVA': 0, 'FUVB': 0},
                    'G130M': {'FUVA': 0, 'FUVB': 0},
                    'G160M': {'FUVA': 0, 'FUVB': 0}}

        for cenwave in df['cenwave'].unique():
            for segment in df['segment'][df['cenwave'] == cenwave].unique():

                sub_df = df[(df['cenwave'] == cenwave) & (df['segment'] == segment)]
                grating = sub_df['opt_elem'].iloc[0]

                # scale to one 
                scaled_df = self.scale_to_1(table=sub_df, size=size)
                binned_wl = np.array([wl for wl in sub_df[f'{size}_binned_wl']])[0]

                best_fit_model = np.array([net for net in scaled_df[f'{size}_best_fit'].iloc[0]])
                best_fit_model_err = np.array([net for net in scaled_df[f'{size}_best_fit_err'].iloc[0]])

                slopes = best_fit_model[:,timebin*2+1]
                slopes_err = best_fit_model_err[:,timebin*2+1]
                
                med = np.median(slopes)
                med_err = np.median(slopes_err)

                indx = np.where((slopes < med+50*med_err)&(slopes > med-50*med_err))

                if np.any(indx): # if no points less than 3 sigma

                    med = np.median(slopes[slopes < 0])
                    indx = np.where(slopes <= 15)

                    if len(binned_wl) <= 2: 

                        if counter_mode[grating][segment] == 0:
                            trace = go.Scatter(
                                x = binned_wl,
                                y = slopes*100.0,
                                error_y = dict(type='data', array=slopes_err*100.0, visible=True),
                                line_color = color_type[grating],
                                marker_symbol=marker_type[grating][segment],
                                mode='markers',
                                name=f'{grating}/{segment}',
                                legendgroup=f'{grating}/{segment}'
                            )
                            counter_mode[grating][segment] = 1
                        else:
                            trace = go.Scatter(
                                x = binned_wl,
                                y = slopes*100.0,
                                error_y = dict(type='data', array=slopes_err*100.0, visible=True),
                                line_color = color_type[grating],
                                marker_symbol=marker_type[grating][segment],
                                mode='markers',
                                name=f'{grating}/{segment}',
                                legendgroup=f'{grating}/{segment}',
                                showlegend=False
                            )

                        fig.add_trace(trace)
                    
                    else:
                        if counter_mode[grating][segment] == 0:
                            trace = go.Scatter(
                                x = binned_wl[indx],
                                y = slopes[indx]*100.0,
                                error_y = dict(type='data', array=slopes_err[indx]*100.0, visible=True),
                                line_color = color_type[grating],
                                marker_symbol=marker_type[grating][segment],
                                mode='markers',
                                name=f'{grating}/{segment}',
                                legendgroup=f'{grating}/{segment}'
                            )
                            counter_mode[grating][segment] = 1
                        else:
                            trace = go.Scatter(
                                x = binned_wl[indx],
                                y = slopes[indx]*100.0,
                                error_y = dict(type='data', array=slopes_err[indx]*100.0, visible=True),
                                line_color = color_type[grating],
                                marker_symbol=marker_type[grating][segment],
                                mode='markers',
                                name=f'{grating}/{segment}',
                                legendgroup=f'{grating}/{segment}',
                                showlegend=False
                            )

                        fig.add_trace(trace)
                
                else:
                    # Needed for the 1327/FUVB bug since mode is no longer monitered
                    if len(slopes[indx]) == 0:
                        continue 
                    if counter_mode[grating][segment] == 0:
                        trace = go.Scatter(
                            x = binned_wl[indx],
                            y = slopes[indx]*100.0,
                            error_y = dict(type='data', array=slopes_err[indx]*100.0, visible=True),
                            line_color = color_type[grating],
                            marker_symbol=marker_type[grating][segment],
                            mode='markers',
                            name=f'{grating}/{segment}',
                            legendgroup=f'{grating}/{segment}'
                        )
                        counter_mode[grating][segment] = 1

                    else:
                        trace = go.Scatter(
                            x = binned_wl[indx],
                            y = slopes[indx]*100.0,
                            error_y = dict(type='data', array=slopes_err[indx]*100.0, visible=True),
                            line_color = color_type[grating],
                            marker_symbol=marker_type[grating][segment],
                            mode='markers',
                            name=f'{grating}/{segment}',
                            legendgroup=f'{grating}/{segment}',
                            showlegend=False
                        )

                    fig.add_trace(trace)

        fig.update_layout(title_text="Slope vs Wavelength")
        fig.update_xaxes(title_text="Wavelength (Å)")
        fig.update_yaxes(title_text="Slope (%/yr)", range=(-20, 5))

        return fig

# --------------------------------------------------------------------------------#
    def add_lines(self, lines, style, name, yrange, second=False):
        """
        """

        # this list will host information about each line
        vlines = []
        if second == False:
            for i, line in enumerate(lines):
                if i == 0:
                    # append plot information of vertical line
                    vlines.append({
                        'x': [line, line],
                        'y': yrange,
                        'mode': 'lines',
                        'line': style,
                        'name': name,
                        'legendgroup': name,
                        'showlegend': True
                    })
                else:
                    vlines.append({
                        'x': [line, line],
                        'y': yrange,
                        'mode': 'lines',
                        'line': style,
                        'name': name,
                        'legendgroup': name,
                        'showlegend': False
                    })
        else:
            for i, line in enumerate(lines):

                vlines.append({
                        'x': [line, line],
                        'y': yrange,
                        'mode': 'lines',
                        'line': style,
                        'name': name,
                        'legendgroup': name,
                        'showlegend': False
                    })
        
        return(vlines)

# --------------------------------------------------------------------------------#
    def get_solar_data(self, spaceweather_url = 'https://www.spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt'):
        """
        Get the solar flux data from this spaceweather website, updates each day.
        """

        # retrieve the flux table from the website where solar flux is hosted
        response = requests.get(spaceweather_url)
        if response.status_code == 200:
            with open('fluxtable.txt', 'wb') as file:
                file.write(response.content)
            with open('fluxtable.txt', 'r') as file:
                lines = file.readlines()
            
            # grab the data from file
            cleaned_data = ''.join([line for line in lines if '---' not in line])

            # turn into dataframe
            df = pd.read_csv(StringIO(cleaned_data), sep="\s+", comment='#')

            # use fluxjulian to index
            df.index = pd.DatetimeIndex(df['fluxjulian'])
            df['date'] = Time(df['fluxjulian'], format='jd').decimalyear

            # drop the columns we don't need and rename what we do need (date and f10.7)
            df = df.drop(columns=['fluxdate', 'fluxtime', 'fluxcarrington', 'fluxadjflux', 'fluxursi'])
            df.rename(columns= {'fluxobsflux':'f10.7'}, inplace=True)
            df = df.reindex(columns=['date', 'f10.7'])

            # only interested in data post 2009.5, after COS launch
            df = df[df['date'] >= 2009.5]

            # return solar flux df
            return(df)

# --------------------------------------------------------------------------------#
    def tds_backout(self, response, wavelength, mjd, opt_elem, aperture, segment, cenwave, tdstab=None):
        """
        """
        #-------#
        # get correction array
        #-------#

        def calculate_drop(time, ref_time, slope, intercept):
            """
            Equation comes from the current ICD-47
            """

            frac_drop = ( (( time - ref_time ) * slope) / (365.25*100) ) + intercept
            return frac_drop
        
        tds_data = fits.getdata(tdstab,ext=1 )
        REF_TIME = fits.getval(tdstab,'REF_TIME',ext=1) #Should be 52922.0 (2003.77)
        mode_index = np.where((tds_data['OPT_ELEM'] == opt_elem) &
                            (tds_data['APERTURE'] == aperture) &
                            (tds_data['SEGMENT'] == segment) &
                            (tds_data['CENWAVE'] == cenwave))[0]

        mode_line = tds_data[ mode_index ]

        tds_nt = mode_line['NT'][0]
        tds_wavelength = mode_line['WAVELENGTH'][0]
        smaller_index = np.where( mode_line['TIME'][0][:tds_nt] < mjd )[0]
        time_index = smaller_index.max()

        tds_slope = mode_line['SLOPE'][0]
        tds_intercept = mode_line['INTERCEPT'][0]

        correction_array = np.zeros( len(tds_wavelength) )
        
        for i, _ in enumerate( tds_wavelength ):
            correction = calculate_drop( mjd, REF_TIME, tds_slope[time_index,i], tds_intercept[time_index,i] )
            correction_array[i] = correction

        #-------
        # interpolate onto input arrays
        #------

        interp_function = interp1d( tds_wavelength, correction_array, 1 )
        interp_correction = interp_function( wavelength )

        return response / interp_correction
# --------------------------------------------------------------------------------#
    def broken_lines(self, x, *p):
        """ Fitting function for a segmented line with n_bp breakpoints."""
        # The actual function that is done.
        # Uses date - reftime and initial parameters
        # and returns the yvalues of all the linear fits
        model_y = np.zeros(len(x))

        # number of breakpoints
        n_bp = 0

        # number of parameters
        n_pars = len(p)

        if n_pars == 1:
            model_y = [p[0] for e in x]
        
        elif n_pars == 2:
            # If there are only two parameters (aka zero breakpoints), then
            # do a single linear fit for each time value of the exposure
            # this will be date - reftime
            y = [p[0] + p[1] * e for e in x]
        
        # If there is an even amount of parameters (as there should), do the the fit
        elif not len(p) % 2:

            # Removes the intercept and the slope (first two elements in p0) and 
            # counts the number of breakpoints we are working with
            n_bp = int((len(p) - 2) // 2)

            # x-values == number of breakpoints
            x_bp = np.zeros(n_bp)

            # y-values == number of breakpoints, scaled net count related
            y_bp = np.zeros(n_bp)

            # set first breakpoint to first value in x array, time-related
            x_bp[0] = p[2]

            # Do linear fit with the first handful of parameters where
            #       p[0]: intercept, initial parameters
            #       p[1]: slope, initial parameters
            #       p[2]: first breakpoint, time-related
            y_bp[0] = p[0] + p[1] * p[2]

            # Loop over the amount of breakpoints, skipping the first breakpoint
            for j in range(1, n_bp):
                # set x value to the next breakpoint
                x_bp[j] = p[2 * (j+1)]

                # Calculate the yvalue of that breakpoint
                #       y_bp[j - 1]: last y value, intercept
                #       p[2 * j + 1]: the next slope value of this breakpoint, slope
                #       x_bp[j] - x_bp[j - 1]: subtract the last breakpoint to the current one to be time-related, time-related x value
                y_bp[j] = y_bp[j - 1] + p[2 * j + 1] * (x_bp[j] - x_bp[j - 1])
            
            # Loop over the number of xvalues, aka number of date values from all exposures
            for i, _ in enumerate(x):

                # Get first in time line segment
                if x[i] < x_bp[0]:

                    # Do a linear fit
                    yy = p[0] + p[1] * x[i]
                
                # Get last in time line segment
                elif x[i] >= x_bp[-1]:

                    # Do a linear fit
                    #       y_bp[-1]: last yvalue breakpoint related, intercept
                    #       p[-1]: last slope value, initial parameters
                    #       (x[i] - p[-2]): last date in time subtracted by last breakpoint in initial parameters
                    yy = y_bp[-1] + p[-1] * (x[i] - p[-2])
                
                # Get the in between line segments
                elif n_bp > 1:

                    # Loop over the breakpoints
                    for j in range(n_bp - 1):

                        # Get the date values in between the breakpoints
                        if (x[i] >= x_bp[j]) and (x[i] < x_bp[j+1]):

                            # Do a linear fit
                            #       y_bp[j]: yvalue of this time period as intercept
                            #       p[2 * (j + 1) + 1]: slope value corresponding to this time segment, slope
                            #       (x[i] - x_bp[j]): date in time subtracted by this breakpoint, time-related
                            yy = y_bp[j] + p[2 * (j + 1) + 1] * (x[i] - x_bp[j])
                            break
                # Put the fitted yvalues into an array
                model_y[i] = yy
            
        else:
            print(f'Warning, number of fit parameters {len(p)}')
            print('is not even, fit may be rubbish.')

        return(model_y)
    
# --------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
# MATH TOOLS
# --------------------------------------------------------------------------------#
# --------------------------------------------------------------------------------#
    def scale_to_1(self, table, size):
        """
        """
        breakpoints = self.breakpoints - self.reftime

        if len(table) == 0:
            return
        
        scaled_to_1_table = {}

        # date of all exposures of cenwave and segment 
        x = np.array(table['date-obs']).flatten() - self.reftime

        binned_net = np.array([binned_net for binned_net in table[f'{size}_scaled_net']])
        stdev = np.array([stdev for stdev in table[f'{size}_scaled_stdev']])

        best_fit      = np.empty(( len(table[f'{size}_binned_wl'].iloc[0]), (len(self.breakpoints)+1)*2))
        best_fit_err  = np.empty(( len(table[f'{size}_binned_wl'].iloc[0]), (len(self.breakpoints)+1)*2))

        for i, _ in enumerate(table[f'{size}_binned_wl'].iloc[0]):
            
            # Scaled net count of given cenwave and segment
            y = binned_net[:,i]
            error = stdev[:,i]

            # Remove NaN values from y-array and from the corresponding xarray and error array
            x = x[~np.isnan(y)]
            error = error[~np.isnan(error)]
            y = y[~np.isnan(y)]

            # initial parameter guess
            initial_params = list(np.polyfit(x, y, 1))

            # polyfit returns the highest order coefficient first
            # begin with intercept then slope [intercept, slope]
            initial_params.reverse()

            for bp in breakpoints:
                # Append breakpoint minus reftime to end of initial parameters list
                initial_params.append(bp)

                # Append slope of that breakpoint (minus reftime) to end of initial parameters list
                initial_params.append(initial_params[1])
            
            # Check to see if cenwave is a bluemode
            blue_flag = False
            if table['cenwave'].iloc[0] in [1222, 1096, 1055]:
                blue_flag = True
            
            # Create the parameter info list which mpfit.mpfit uses
            parinfo = self.create_parainfo_list(initial_params, x, blue_flag)

            # conduct the fit here
            fit = self.mpfit_the_data(self.mpfitting_function, x, y, err = error, p0=initial_params, parinfo=parinfo)

            pars = fit[0]
            perrs = fit[2]

            best_fit[i,:] = pars
            best_fit_err[i,:] = perrs

            if pars is None:
                print('No fit was done.')
            else:
                y_intercept = best_fit[i,0]
                y_inter_stdev = best_fit_err[i,0]

                # Scale to 1
                stdev[:,i] = stdev[:,i]/y_intercept
                binned_net[:,i] = binned_net[:,i]/y_intercept
                best_fit[i, 1::2] = best_fit[i, 1::2]/y_intercept

                best_fit_err[i, 1::2] = np.sqrt(
                    (best_fit_err[i, 1::2] / y_intercept)**2 +
                    (best_fit[i, 1::2] / (y_intercept**2)*y_inter_stdev)**2
                )

                best_fit[i, 0] = 1.0

        # Set the appropriate values to dataframe
        scaled_to_1_table[f'{size}_scaled_net'] = [binned_net]
        scaled_to_1_table[f'{size}_scaled_stdev'] = [stdev]
        scaled_to_1_table[f'{size}_best_fit'] = [best_fit]
        scaled_to_1_table[f'{size}_best_fit_err'] = [best_fit_err]

        scaled_to_1_table = pd.DataFrame(scaled_to_1_table)
        
        return(scaled_to_1_table)

# --------------------------------------------------------------------------------#

    def create_parainfo_list(self, p0, x, blue_flag):
        """
        Create the parinfo list of dictionaries for mp

        Args:
            p0: list
                a list of best guess input parameters
            x: array
                an array of dates of exposures minus reftime
            blue_flag: boolean
                If the cenwave in particular is a bluemode
        
        Returns:
            parinfo: list
                A dictionary of input parameters to pass into mpfit.
                This is used to fix the breakpoint values (not try to find the best fit)
        """

        # Create array the length of initial parameters list
        flag = np.zeros(len(p0), dtype=int)

        # Create array the length of all breakpoints, will contain all breakpoints minus one
        bp = np.zeros((len(p0)-2)//2)

        # Set last flag as one to be an edge/boundary
        flag[-1] = 1

        # Loop over length of initial parameters list starting at index 2 for 2-stepsizes
        #OR loop over the amount of breakpoints minus one.
        for j in range(2, len(p0), 2):

            # Set breakpoint value from initial parameters into its own array
            bp[(j-2)//2] = p0[j]

            # Set flag index = 1 if the breakpoint in initial_parameters is greater or equal
            # to first in time date (x)
            if p0[j] >= x[0]:
                flag[j] = 1 # Flag index corresponding the breakpoint index in initial paramters
                flag[j-1] = 1 # Flag index before breakpoint index in initial parameters
                flag[j+1] = 1 # Flag index after breakpoint index in initial paramters
            
            # Get the index of the first slope that will be fitted
            indx = np.where(flag == 1)[0][0] # index of the first slope to be fitted
            tied = f'p[{str(indx)}]'

            # Do not fit slope of first segment if less than 4 points for blue mode
            if (x[0] < bp[-1]) and (blue_flag == True):
                
                # If there are less than 4 points for blue mode in first in time then set
                # flag in those indx to be -1 (to be excluded) and look at next two indx
                if ( len(np.where(x < bp[(indx-1)//2])[0]) < 4):
                    flag[indx] = -1
                    tied = f'p[{str(indx+2)}]'
            
            # List that will hold all the dictionaries
            parinfo = []

            # Loop over length of initial paramters list
            for j, _ in enumerate(p0):

                # Fix the breakpoints
                if (j > 1) and (j%2 == 0):
                    parinfo.append({
                        'value': p0[j], # breakpoint - reftime
                        'fixed': True,
                        'limited':[False, False],
                        'limits': [0,0],
                        'step': None,
                        'mpside': 0,
                        'tied':'',
                        'mpprint': 1
                    })
                else:
                    if (j > 0) and (flag[j] == 0):

                        # Fix slopes of segment with no data to slope of first segment with data
                        # (in practice, assumes all segments have data if breakpoint is after the 1st data)
                        p0[j] = 0.0
                        parinfo.append({
                            'value': p0[j],
                            'fixed': True,
                            'limited': [False, False],
                            'limits': [0,0],
                            'step': None,
                            'mpside': 0,
                            'mpmaxstep': 0,
                            'tied': '',
                            'mpprint': 1
                        })
                    
                    elif (flag[j] == -1):

                        # For blue modes, ties slope of 1st segment with data to slope of the 2nd segment
                        # if the 1st segment has less than 4 measurements (fit results were unstable from one
                        # wavelength bin to another)
                        parinfo.append({
                            'value': p0[indx],
                            'fixed': True,
                            'limited': [False, False],
                            'limits': [0,0],
                            'step': None,
                            'mpside': 0,
                            'mpmaxstep': 0,
                            'tied': tied,
                            'mpprint': 1
                        })

                    else:
                        parinfo.append({
                            'value': p0[j],
                            'fixed': False,
                            'limited': [False, False],
                            'limits': [0,0],
                            'step': None,
                            'mpside': 0,
                            'mpmaxstep': 0,
                            'tied': '',
                            'mpprint': 1
                        })
        return (parinfo)
    
# --------------------------------------------------------------------------------#
    def mpfit_the_data(self, func, x, y, err=None, p0=None, parinfo=None):
        """
        Call the fitting function. Uses mpfit. 

        Args: 
            func: function
                mpfit function
            x: array
                date - reftime
            y: array
                scaled net count rate
            err: array
                stdev of scaled net count rate
            p0: list
                initial parameters guess
            parinfo: list of dictionaries
                list of dictionaries with parameter information used in mpfit
        """
        if err is None:
            err = np.ones(len(x))
        if p0 is None:
            print('error, initial parameter guesses not provided')
            return (None, None, None)
        if len(p0) == 1: # Zero line segments.
            w = 1.0 / err**2
            """
            Return the weighted mean, unbiased weighted sample variance, and
            the weighted standard error.
            """
            if len(y) <= 0: #There is no data.
                return (None, None, None)
            elif len(y) == 1: # No variance or standard error
                return (y[0], 0.0,0.0)
            
            # Weighted mean.
            wmeany, wsum = np.ma.average(y, weights=w, returned=True)

            # Unbiased weighted sample variance
            wsum2 = wsum**2
            w2sum = np.sum(w**2)
            n = len(w)

            if abs(wsum2 - w2sum) < np.finfo(np.dtype(wsum2)).eps:
                s2 = 1.0 / (n - 1) * np.sum((y - wmeany)**2)
            else:
                s2 = (np.sum(w * y**2) * wsum - np.sum(w * y)**2) / (wsum2 - w2sum)

            # Standard error in the weighted mean.
            t1 = n / ((n - 1) * wsum2)
            meanw = np.mean(w)
            t2 = np.sum((w * x - meanw * wmeany)**2)
            t3 = wmeany * np.sum((w - meanw) * (w * y - meanw * wmeany))
            t4 = wmeany**2 * np.sum((w - meanw)**2)
            wse = np.sqrt(t1 * (t2 - 2.0 * t3 + t4))

            return ([wmeany], [s2], [wse])
        
        fa = {'x': x, 'y': y, 'err': err}

        m = mpfit.mpfit(self.mpfitting_function, p0, functkw=fa, parinfo=parinfo, quiet=True, debug=False)

        if m.status <= 0:
            print(f'mpfit, status={m.status}, {m.errmsg}')
            popt, pcov, perr = None, None, None
        
        # Grab the parameters, covarience, and parameters error
        popt = m.params
        pcov = m.covar
        perr = m.perror

        return (popt, pcov, perr)

# --------------------------------------------------------------------------------#
    def mpfitting_function(self, p, fjac=None, x=None, y=None, err=None):
        """
        Mpfit fitting function.

        Args:
            p: list
                list of initial parameters, p0
            fjac: unknown
                partial derivatives?
            x: array
                date array of date of exposures - reftime in decimal year
            y: array
                scaled net count rate
            err: array
                scaled stdev of scaled net count rate
        """

        status = 0
        if fjac is not None:
            print('mpfitting_function does not calculate partial derivatives.')
            status = -1
        if x is None:
            print('mpfitting_function requires that x be provided.')
            status = -2
        if y is None:
            print('mpfitting_function requires that y be provided.')
            status = -2
        if err is None:
            print('The error vector is assumed to be unity.')
            err = np.ones(len(x))
        elif 0.0 in err[:]:
            print('Zero value in error vector.')
            status = -3
        
        if status == 0:
            model_y = self.broken_lines(x, *p)

        weighted_deviation = (y - model_y) / err

        return ([status, weighted_deviation])
