from astropy.io import fits, ascii
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

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from urllib import request
from astropy.convolution import Box1DKernel, convolve

"""
This is the base class for the FUVTDS Monitor that will do all
the analysis need and provide the necessary component to the
monitor that will conduct all the plotting.

Please be aware, this will be a monster.

Notes to do in the future:
    - Combine get hdu info with bin data
    - Exchange mpfit.mpfit with scipy.curve_fit
    - Do multithreading to run each cenwave/segment combo in parallel, for analysis
    - Put outputs into a log file instead out outright outputting
    - Fix the bug that affects the LP3 -> LP4 -> LP3, present in 1280 FUVB
    - Fix errors bug
    - Add in a function for 'broken lines' that can be called in for the plotting class
    - add highlighted areas in residual plots in plotting class for 2% and 5%
    - add in current tdstab as a class input in plotting function
"""

__author__ = 'J. Hernandez' #Me! JAQ!

class FUVTDSBase:
    """
    This class will analyze and store the results necessary to
    conduct a FUVTDS monitor analysis.

    Attributes:
        cenwaves (list): The central wavelength modes used in routine FUV TDS monitoring.
        breakpoints (array-like): All the TDS breakpoints by fractional year.
        reftime (float.64): The decimal year of the reftime.
        small / large (dictionary): A nestled dictionary made from the small wavelength-bins,
                            organized by cenwave and then segment. ie small[1533][FUVA]
            binned_net (array-like): binned NET array for each x1d file, per segment.
            binned_wl (array-like): binned WAVELENGTH array for each x1d file, per segment.
            stdev (array-like): binned standard deviation of the NET array binning.
            grating (array-like): OPT_ELEM keyword for each x1d file, per segment.
            lp (array-like): LIFE_ADJ keyword for each x1d file, per segment.
            target (array-like): TARGNAME keyword for each x1d file, per segment.
            rootname (array-like): Rootnames of all input x1d files used for that cenwave and segment.
            date (array-like): Decimal year date of the time the exposure was taken for that cenwave and segment.
            infiles (array-like): x1d files for that cenwave and segment, organized chronologically.
            scaled_net (array-like): binned NET array for each x1d file of each cenwave and segment, scaled between
                                    all LPs involved and scaled to 1.
            scaled_stdev (array-like): binned standard deviation of the NEET array binning for each x1d file of 
                                    each cenwave and segment, scaled between all LPs involved and scaled to 1.
            scale_factor (array-like): the scale factor used to scale binned NET array of each x1d file of each
                                    cenwave and segment across all LPs involved.
    """

    def __init__(self, PIDs, reftime = 54952.0, inventory='inventory_test.csv'):
        """
        Args:
            PIDs: The dat file that will store all the PIDs part of the
                FUVTDS Monitor programs.
            reftime: The mjd of the reftime.
            breakpoints: All the TDS breakpoints by fractional year.
            inventory: The csv file containing the x1d files and other information used in the FUV TDS Monitor
        """
        self.breakpoints = np.array([2010.2, 2011.2, 2011.75, 2012.0, 2012.8, 2013.8, 2015.5, 2019.0, 2020.6, 2022.0, 2023.2])
        self.HV_FUVA = np.array([2012.23,2012.56,2014.84,2015.107,2017.75,2020.75,2021.76, 2023.94])
        self.HV_FUVB = np.array([2011.18,2013.47,2012.56,2014.55,2015.107,2016.05,2017.75,2020.75,2022.47, 2023.94])
        self.LPs = np.array([2012.56, 2015.107, 2017.75, 2021.76, 2022.75])
        self.reftime = Time(reftime, format="mjd").decimalyear

        tables = self.parse_infiles(PIDs, inventory)
        tables = self.scalings(tables)

        # scale all to one
        #self.small = self.scale_to_1_all_data(small)
        #self.large = self.scale_to_1_all_data(large)
            
# --------------------------------------------------------------------------------#
    def scale_to_1_all_data(self, dictionary):
        """
        Scale all the monitored modes to a 1, where 1 is the reference date of when COS began
        observing. Through this we can see how each monitored mode is losing sensitivity over
        time. 

        Note: for the sake of things, I will use the original mpfit.mpfit package and on a later date
        I will see if I can trade it out for scipy curvefit. There should not be any significant changes.
        Though I'm not entirely sure how the mpfit.mpfit works  

        Args:
            Dictionary: the dictionary already scalled between all lps
        
        Returns: 
            dictionary: dictionary that contains all monitored modes now scaled to 1.


        """

        def create_parainfo_list(p0, x, blue_flag):
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
        
        def mpfit_the_data(func, x, y, err=None, p0=None, parinfo=None):
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

            m = mpfit.mpfit(mpfitting_function, p0, functkw=fa, parinfo=parinfo, quiet=True, debug=False)

            if m.status <= 0:
                print(f'mpfit, status={m.status}, {m.errmsg}')
                popt, pcov, perr = None, None, None
            
            # Grab the parameters, covarience, and parameters error
            popt = m.params
            pcov = m.covar
            perr = m.perror

            return (popt, pcov, perr)
        
        # Fitting function that will be used
        def mpfitting_function(p, fjac=None, x=None, y=None, err=None):
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

                    # Removees the intercept and the slope (first two elements in p0) and 
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

            weighted_deviation = (y - model_y) / err

            return ([status, weighted_deviation])

        breakpoints = self.breakpoints - self.reftime

        for cenwave in dictionary:
            for segment in dictionary[cenwave]:
                for i, _ in enumerate(dictionary[cenwave][segment]['binned_wl']):

                    # Date of all exposures of cenwave and segment minus ref time
                    x = dictionary[cenwave][segment]['date'] - self.reftime
                    
                    # Scaled net count of given cenwave and segment
                    y = dictionary[cenwave][segment]['scaled_net'][:, i].ravel()

                    # Scaled stdev of given cenwave and segment
                    error = dictionary[cenwave][segment]['scaled_stdev'][:,i].ravel()

                    # Remove NaN values from y-array and from the corresponding xarray and error array
                    x = x[~np.isnan(y)]
                    error = error[~np.isnan(y)]
                    y = y[~np.isnan(y)]

                    # initial parameter guess
                    initial_params = list(np.polyfit(x, y, 1))

                    # polyfit returns the highest order coefficient first
                    # Begin with intercept then slope [intercept, slope]
                    initial_params.reverse()

                    # loop for amount of breakpoints
                    for bp in breakpoints:
                        
                        # Append breakpoint minus reftime to end of initial parameters list
                        initial_params.append(bp)

                        # Append slope of that breakpoint (minus reftime) to end of initial parameters list
                        initial_params.append(initial_params[1])
                    
                    # Check to see if cenwave is a bluemode
                    blue_flag = False
                    if cenwave in [1222, 1096, 1055]:
                        blue_flag = True
                    
                    # Create the parameter info list which mpfit.mpfit uses
                    parinfo = create_parainfo_list(initial_params, x, blue_flag)

                    # conduct the fit here
                    fit = mpfit_the_data(mpfitting_function, x, y, err = error, p0 = initial_params, parinfo = parinfo)

                    # The fit values
                    pars = fit[0]
                    perrs = fit[2]

                    dictionary[cenwave][segment]['best_fit'][i,:] = pars
                    dictionary[cenwave][segment]['best_fit_err'][i, :] = perrs

                    if pars is None:
                        print('No fit was done.')
                    else:
                        y_intercept = dictionary[cenwave][segment]['best_fit'][i, 0]
                        y_inter_stdev = dictionary[cenwave][segment]['best_fit_err'][i, 0]

                        # Scale to 1
                        dictionary[cenwave][segment]['scaled_stdev'][:, i] = dictionary[cenwave][segment]['scaled_stdev'][:,i]/y_intercept
                        dictionary[cenwave][segment]['scaled_net'][:, i] = dictionary[cenwave][segment]['scaled_net'][:,i]/y_intercept
                        dictionary[cenwave][segment]['best_fit'][i, 1::2] = dictionary[cenwave][segment]['best_fit'][i, 1::2]/y_intercept

                        dictionary[cenwave][segment]['best_fit_err'][i, 1::2] = np.sqrt(
                            (dictionary[cenwave][segment]['best_fit_err'][i, 1::2] / y_intercept)**2 +
                            (dictionary[cenwave][segment]['best_fit'][i, 1::2] / (y_intercept**2)*y_inter_stdev)**2
                        )
                        dictionary[cenwave][segment]['best_fit'][i, 0] = 1.0
        
        return(dictionary)

# --------------------------------------------------------------------------------#
    def scalings(self, table):
        """
        """

        for segment in set(table['segment']):
            with mp.Pool(16) as pool:
                pool.starmap(self.scale_lps, zip(set(table['cenwave']), repeat(segment), repeat(table)))
            pool.terminate
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
                    
                    scale_factor = (scaled_to[f'{size}_binned_net'].iloc[a][i]/
                                    scaled[f'{size}_binned_net'].iloc[0][i])
                    
                    for j, _ in enumerate(scaled[f'{size}_binned_net']):
                        scaled[f'{size}_binned_net'].iloc[j][i] = (
                            scaled[f'{size}_binned_net'].iloc[j][i]*
                            scale_factor
                        )
                
            return(scaled)
        
        table = table[(table['cenwave'] == cenwave) & (table['segment'] == segment)]

        if len(table) == 0:
            return
        
        new_table = []
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
            new_table.append(lp6)
            print(f"+++ Scaling LP6 to LP4 using data from datasets: {lp4['file_path'].iloc[a]} {lp6['file_path'].iloc[0]}")

        if (5 in table['life_adj']) & (4 in table['life_adj']):

            lp5_indx_wd308 = np.where((table['life_adj'] == 5) &
                                    (table['targname'] == 'WD0308-565'))
            lp5_indx_gd71 = np.where((table['life_adj'] == 5) &
                                    (table['targname'] == 'GD71'))
            
            lp4_indx_wd308 = np.where((table['life_adj'] == 4) &
                                    (table['targname'] == 'WD0308-565'))
            lp4_indx_gd71 = np.where((table['life_adj'] == 4) &
                                    (table['targname'] == 'GD71'))
            
            if (cenwave > 1500) & (segment == 'FUVA'):
                lp4_indx = lp4_indx_gd71
                lp5_indx = lp5_indx_gd71
            else: 
                lp4_indx = lp4_indx_wd308
                lp5_indx = lp5_indx_wd308

            lp4_indx = lp4_indx[0]
            lp5_indx = lp5_indx[0]

            a = find_nearest(table['date-obs'][lp4_indx], 
                            table['date-obs'][lp5_indx[0]])
            
            for i, _ in enumerate(table['binned_wl']):
                #Scale LP5 data
                table['scale_factor'][lp5_indx, i] = (
                    table['binned_net'][lp4_indx[a], i] /
                    table['binned_net'][lp5_indx[0], i]
                )
                table['scaled_net'][lp5_indx, i] = (
                    table['scaled_net'][lp5_indx, i] *
                    table['scale_factor'][lp5_indx, i]
                )
                
                # calculate error
                table['scaled_stdev'][lp5_indx, i] = calc_error(
                    i, table, lp4_indx, lp5_indx
                )
            print(f"+++ Scaling LP5 to LP4 using data from datasets: {table['infiles'][lp4_indx[a]]} {table['infiles'][lp5_indx[0]]}")
        


        if (4 in table['life_adj']) & (3 in table['life_adj']):
            lp4_indx_wd308 = np.where((table['life_adj'] >= 4) &
                                    (table['targname'] == 'WD0308-565'))
            lp4_indx_gd71 = np.where((table['life_adj'] >= 4) &
                                    (table['targname'] == 'GD71'))
            
            lp3_indx1_wd308 = np.where((table['life_adj'] == 3) &
                                    (table['targname'] == 'WD0308-565') &
                                    (table['date-obs'] < 2019.0))
            lp3_indx1_gd71 = np.where((table['life_adj'] == 3) &
                                    (table['targname'] == 'GD71')&
                                    (table['date-obs'] < 2019.0))
            
            lp3_indx2_wd308 = np.where((table['life_adj'] == 3) &
                                    (table['targname'] == 'WD0308-565') &
                                    (table['date-obs'] > 2019.0))
            lp3_indx2_gd71 = np.where((table['life_adj'] == 3) &
                                    (table['targname'] == 'GD71')&
                                    (table['date-obs'] > 2019.0))
            

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

            for i, _ in enumerate(table['binned_wl']):
                #Scale LP4 and LP3 after LP4->LP3
                if cenwave != 800:
                    table['scale_factor'][lp4_indx, i] = (
                        table['binned_net'][lp4_indx[0]-1, i] /
                        table['binned_net'][lp4_indx[0], i]
                    )
                    table['scaled_net'][lp4_indx, i] = (
                        table['scaled_net'][lp4_indx, i] *
                        table['scale_factor'][lp4_indx, i]
                    )
                    # calculate error
                    table['scaled_stdev'][lp4_indx, i] = calc_error(
                        i, table, lp3_indx1, lp4_indx
                    )
                    print(f"+++ Scaling LP4 to LP3 using data from datasets: {table['infiles'][lp4_indx[0]-1]} {table['infiles'][lp4_indx[0]]}")

                    # Scale LP3 after LP4 -> LP3
                    if len(lp3_indx2) > 0:
                        table['scale_factor'][lp3_indx2, i] = (
                            table['binned_net'][lp4_indx[0], i] / 
                            table['binned_net'][lp4_indx[0]-1, i])
                        
                        table['scaled_net'][lp3_indx2, i] = (
                            table['scaled_net'][lp3_indx2, i] * 
                            table['scale_factor'][lp3_indx2, i])

                        # calculate error
                        table['scaled_stdev'][lp3_indx2, i] = calc_error(
                            i, table, lp4_indx, lp3_indx2
                        )
                        print ('+++ Scaling LP3 to LP4 using data from datasets: ', table['infiles'][lp4_indx[0]],table['infiles'][lp4_indx[0]-1])
                elif cenwave == 800:
                    table['scale_factor'][lp3_indx2, i] = (table['binned_net'][lp3_indx2[0]+1, i] / table['binned_net'][lp3_indx2[0], i])
                    table['scaled_net'][lp3_indx2, i] = table['scaled_net'][lp3_indx2, i] * table['scale_factor'][lp3_indx2, i]
                    # calculate error
                    table['scaled_stdev'][lp3_indx2, i] = calc_error(
                        i, table, lp4_indx, lp3_indx2
                    )

                    print ('+++ Scaling LP3 to LP4 using data from datasets: ', table['infiles'][lp3_indx2[0]+1], table['infiles'][lp3_indx2[0]])

        if (3 in table['life_adj']) & (2 in table['life_adj']):

            lp3_indx_wd308 = np.where((table['life_adj'] >= 3) &
                                    (table['targname'] == 'WD0308-565'))
            lp3_indx_gd71 = np.where((table['life_adj'] >= 3) &
                                    (table['targname'] == 'GD71'))
            
            lp2_indx_wd308 = np.where((table['life_adj'] == 2) &
                                    (table['targname'] == 'WD0308-565'))
            lp2_indx_gd71 = np.where((table['life_adj'] == 2) &
                                    (table['targname'] == 'GD71'))
            
            if (cenwave > 1500) & (segment == 'FUVA'):
                lp2_indx = lp2_indx_gd71
                lp3_indx  = lp3_indx_gd71
            else: 
                lp2_indx = lp2_indx_wd308
                lp3_indx  = lp3_indx_wd308
            
            lp2_indx = lp2_indx[0]
            lp3_indx  = lp3_indx[0]

            for i, _ in enumerate(table['binned_wl']):
                # Scale LP3
                table['scale_factor'][lp3_indx, i] = (
                    table['binned_net'][lp2_indx[-1], i] /
                    table['binned_net'][lp3_indx[0], i]
                )
                table['scaled_net'][lp3_indx, i] = (
                    table['scaled_net'][lp3_indx, i] *
                    table['scale_factor'][lp3_indx, i]
                )
                
                # calculate error
                table['scaled_stdev'][lp3_indx, i] = calc_error(
                    i, table, lp2_indx, lp3_indx
                )
            print(f"+++ Scaling LP3 to LP2 using data from datasets: {table['infiles'][lp2_indx[-1]]} {table['infiles'][lp3_indx[0]]}")
        

        if (2 in table['life_adj']) & (1 in table['life_adj']):
            lp2_indx_wd308 = np.where(
                (table['life_adj'] >= 2) &
                (table['targname'] == 'WD0308-565'))
            lp2_indx_gd71 = np.where(
                (table['life_adj'] >= 2) &
                (table['targname'] == 'GD71'))
            

            lp1_indx_wd1057 = np.where(
                (table['life_adj'] == 1) &
                (table['targname'] == 'WD1057+719'))
            lp1_indx_wd0947 = np.where(
                (table['life_adj'] == 1) &
                (table['targname'] == 'WD0947+857'))
            
            
            if (cenwave > 1500) & (segment == 'FUVA'):
                lp1_indx = lp1_indx_wd1057
                lp2_indx = lp2_indx_gd71
            elif (cenwave > 1500) & (segment == 'FUVB'):
                lp1_indx = lp1_indx_wd1057
                lp2_indx = lp2_indx_wd308
            else: 
                lp1_indx = lp1_indx_wd0947
                lp2_indx  = lp2_indx_wd308
            
            lp1_indx = lp1_indx[0]
            lp2_indx = lp2_indx[0]

            for i, _ in enumerate(table['binned_wl']):
                # Scale LP2
                table['scale_factor'][lp2_indx, i] = (
                    table['binned_net'][lp1_indx[-1], i] /
                    table['binned_net'][lp2_indx[0], i]
                )
                table['scaled_net'][lp2_indx, i] = (
                    table['scaled_net'][lp2_indx, i] *
                    table['scale_factor'][lp2_indx, i]
                )
                
                # calculate error
                table['scaled_stdev'][lp2_indx, i] = calc_error(
                    i, table, lp1_indx, lp2_indx
                )
            print(f"+++ Scaling LP2 to LP1 using data from datasets: {table['infiles'][lp1_indx[-1]]} {table['infiles'][lp2_indx[0]]}")

        return (table)

# --------------------------------------------------------------------------------#
    def hdu_bin(self, file):
        """
        """

        targ_info_dict = {
            1533: {'FUVA': ['GD71'], 'FUVB': ['WD0308-565']},
            1577: {'FUVA': ['GD71', 'WD1057+719'], 'FUVB': ['WD1057+719', 'WD0308-565']},
            1623: {'FUVA': ['GD71', 'WD1057+719'], 'FUVB': ['WD1057+719', 'WD0308-565']},
            1291: {'FUVA': ['WD0308-565', 'WD0947+857'], 'FUVB': ['WD0308-565', 'WD0947+857']},
            1327: {'FUVA': ['WD0308-565', 'WD0947+857'], 'FUVB': ['WD0308-565', 'WD0947+857']},
            1105: {'FUVA': ['WD0308-565', 'WD0947+857']},
            1280: {'FUVA': ['WD0308-565', 'WD0947+857'], 'FUVB': ['WD0308-565', 'WD0947+857']},
            800:  {'FUVA': ['WD0308-565']},
            1222: {'FUVA': ['WD0308-565'], 'FUVB': ['WD0308-565']},
            1055: {'FUVA': ['WD0308-565'], 'FUVB': ['WD0308-565']},
            1096: {'FUVB': ['GD71']}
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

                    small_x_index = np.where((wl >= small_min_wl) & (wl <= small_max_wl))
                    large_x_index = np.where((wl >= large_min_wl) & (wl <= large_max_wl))

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
                        np.std, bins=small_bins
                    )[0]
                    large_stdev = binned_statistic(
                        wl[large_x_index],
                        net[large_x_index],
                        np.std, bins=large_bins
                    )[0]

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
                            'exptime': [hdr1['exptime']],
                            'file_path': [file],
                            'small_binned_net': [small_mean_net],
                            'small_stdev': [small_stdev],
                            'small_binned_wl': [small_edges[:-1]+np.diff(small_edges)/2],
                            'small_wl_edges': [small_edges],
                            'large_binned_net': [large_mean_net],
                            'large_stdev': [large_stdev],
                            'large_binned_wl': [large_edges[:-1]+np.diff(large_edges)/2],
                            'large_wl_edges': [large_edges]
                            }
                        )
                    tables.append(x1d_table)
                table = pd.concat(tables)

                return (table)

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
           tables = tables.sort_values(by=['date-obs'], ignore_index=True)
           
           csv_file_save = tables.drop(columns=['small_binned_net', 
                                                'small_binned_wl', 
                                                'small_wl_edges',
                                                'large_binned_net', 
                                                'large_binned_wl', 
                                                'large_wl_edges'
                                                ])
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
            all_programs: the PIDs of all the programs used in the FUVTDS monitor
            pattern: the pattern of file we want, in this case x1d files.
        Returns:
            path_list: the path to the x1d file.
        """
        total_path = os.path.join(COSMO, all_programs, pattern)
        path_list = glob.glob(total_path)
        return(path_list)

class FUVTDSMonitor(object):
    """
    This plotting function
    """

    # adding the life time positions, hv raises, and whatever else lol


    def __init__(self, TDSData) -> None:
        """
        Args:
            TDSData: object that contains all the relavent information of the TDS analysis.
        """
        self.trends = TDSData

        # the vertical lines used in plots
        self.breakpoints = TDSData.breakpoints
        self.HV_FUVA = TDSData.HV_FUVA
        self.HV_FUVB = TDSData.HV_FUVB
        self.LPs = TDSData.LPs

        # plotting
        self.solar  = self.get_solar_data()
        self.plot_solar_flux()
        self.rel_sens()
    
    # add vertical lines
    # REVAMP THIS THIS DOESNT REALLY WORK AHHH          
    def _add_lines(self, lines, style, name):
        
        # this list will host information about each line
        vlines = []
        for i, line in enumerate(lines):
            if i == 0:
                # Append plot information of vertical line
                vlines.append(go.Scatter(
                x = [line, line],
                y = [0, 1.4],
                mode = 'lines',
                line=style,
                name=name,
                legendgroup=name,
                showlegend=True
            ))
            else:
                # Group together the lines and have only the first visible in legend
                vlines.append(go.Scatter(
                    x = [line, line],
                    y = [0, 1.4],
                    mode = 'lines',
                    line=style,
                    name=name,
                    legendgroup=name,
                    showlegend=False
                ))
        return (vlines)
        
    def get_solar_data(self):
        """
        Get the solar flux data from this spaceweather website, updates each day.
        """

        # website where solar flux is hosted
        spaceweather_url = 'ftp://ftp.seismo.nrcan.gc.ca/spaceweather/solar_flux/daily_flux_values/fluxtable.txt'

        # retrieve the flux table from the website where solar flux is hosted
        request.urlretrieve(spaceweather_url, 'fluxtable.txt')

        # read in the flux table and convert to pandas dataframe
        data = ascii.read('fluxtable.txt')
        df = data.to_pandas()

        # obtain the julian flux column, reindex data for flux julian, and then create a new date column in decimal year
        fluxjulian = df['fluxjulian']
        df.index = pd.DatetimeIndex(fluxjulian)
        df['date'] = Time(fluxjulian, format='jd').decimalyear

        # remove the unnecessary columns, rename the flux column to f10.7, and reindex to date
        df = df.drop(columns=['fluxdate', 'fluxtime', 'fluxcarrington', 'fluxadjflux', 'fluxursi'])
        df.rename(columns = {'fluxobsflux':'f10.7'}, inplace = True)
        df = df.reindex(columns=['date', 'f10.7'])

        # this removes any previous solar flux txt that is out of date
        os.system('rm '+os.path.join("solar_flux.txt"))

        # save the most up to date solar flux file to directory
        outfile = os.path.join("solar_flux.txt")
        df = df[df['date'] >= 2009.5] # only interested in data post 2009.5, after COS launch
        df.to_csv(outfile, header=None, index=None, sep=' ', mode='a')

        return(df)
    
    def plot_solar_flux(self):
        """
        Plot the solar flux against the relative sensitivity of all the modes, large bins.
        """

        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Plot the solar flux from dataframe, both smoothed and unsmoothed
        unsmoothed = go.Scatter(x=self.solar['date'],
                            y=self.solar['f10.7'],
                            line_shape='linear',
                            line=dict(color='royalblue', width=4),
                            name='10.7 cm radio flux',
                            opacity=0.5)
        
        smoothed = go.Scatter(x=self.solar['date'],
                            y=convolve(self.solar['f10.7'], Box1DKernel(150), boundary='extend'),
                            line_shape='linear',
                            line=dict(color='firebrick', width=2),
                            name='Smoothed 10.7 cm radio flux',
                            opacity=0.6)
        
        # Plot the fractional throughput!
        marker_type = {'FUVA': 'circle',
                       'FUVB': 'x'}
        for cenwave in self.trends.large:
            for segment in self.trends.large[cenwave]:

                grating = self.trends.large[cenwave][segment]['grating'][0]

                # Fractional throughput, looped over all monitored modes
                data = go.Scatter(
                    x=self.trends.large[cenwave][segment]['date'],
                    y=self.trends.large[cenwave][segment]['scaled_net'][:,0],
                    mode='markers',
                    marker_symbol=marker_type[segment],
                    name=f'{grating}/{cenwave}/{segment}',
                    customdata= np.stack(
                        (self.trends.large[cenwave][segment]['rootname'],
                         self.trends.large[cenwave][segment]['lp'],
                         self.trends.large[cenwave][segment]['PID'],
                         self.trends.large[cenwave][segment]['target']), 
                         axis=-1
                    ),
                    hovertemplate=
                    'Rootname: %{customdata[0]}<br>'+
                    'Life_adj: %{customdata[1]}<br>'+
                    'Proposid: %{customdata[2]}<br>'+
                    'Target: %{customdata[3]}'
                    "<extra></extra>"
                )

                # add trace to plot
                fig.add_trace(data, secondary_y=False)
        
        # add solar flux smoothed and unsmoothed to plot based off secondary_y
        fig.add_trace(unsmoothed, secondary_y=True)
        fig.add_trace(smoothed, secondary_y=True)

        # add vertical lines
        # this might change bc of relative sensitivity function
        fig.add_traces(self._add_lines(self.breakpoints, dict(color='red', width=2, dash='dash'), 'Breakpoint'))
        fig.add_traces(self._add_lines(self.HV_FUVA, dict(color='purple', width=2, dash='dash'), 'Voltage Change SegA'))
        fig.add_traces(self._add_lines(self.HV_FUVB, dict(color='grey', width=2, dash='dash'), 'Voltage Change SegB'))
        fig.add_traces(self._add_lines(self.LPs, dict(color='grey', width=2, dash='dot'), 'LP switch'))

        # add the title
        fig.update_layout(
            title_text="TDS Solar Flux"
        )

        # set x-axis title
        fig.update_xaxes(title_text="Date")
        
        # set y-axes titles
        fig.update_yaxes(title_text="Fractional Throughput", range=(0.0, 1.1), secondary_y=False)
        fig.update_yaxes(title_text="10.7 cm Flux (units here)", range=(50, 400), secondary_y=True)
        fig.write_html('tds_solar_flux.html') #maybe add the date?
    
    def rel_sens(self):
        """
        Plot the relative sensitivity of all the monitored modes and residuals against best fit
        parameters and current tdstab.
        
        need to add:
        -residuals
        -fitted lines
        -tdstab current
        -verticcal lines
        """

        # large wavelength bins and small wavelength bins are both plotted, will loop over
        sizes = [self.trends.large, self.trends.small]
        
        # for the html file to save as separate files
        name = ['large', 'small']
        
        # loop over the trend sizes
        for j, trends in enumerate(sizes):

            # hold the labels of each unique monitored mode
            labels = []

            # save the total wavelength bins used in all modes
            tot_wl_bins = []

            # create figure with two subplots
            fig = make_subplots(rows=2, cols=1,
                            shared_xaxes=True,
                            vertical_spacing=0.1,
                            subplot_titles=("Original Data", "Residuals"))
            
            # Loop over the cenwave, segment, and wavelength bins
            for cenwave in trends:
                for segment in trends[cenwave]:
                    for i, _ in enumerate(trends[cenwave][segment]['binned_wl']):

                        # Make trace of relative sensntivity of this wavelength bin
                        trace = go.Scatter(
                            x=trends[cenwave][segment]['date'],
                            y=trends[cenwave][segment]['scaled_net'][:,i],
                            error_y=dict(
                                    type='data',
                                    array=trends[cenwave][segment]['scaled_stdev'][:,i],
                                    visible=True
                                ),
                            mode='markers',
                            name=f"{trends[cenwave][segment]['grating'][0]}/{cenwave}/{segment} {trends[cenwave][segment]['wl_bin_edges'][i]} - {trends[cenwave][segment]['wl_bin_edges'][i+1]}",
                            customdata= np.stack(
                                    (trends[cenwave][segment]['rootname'],
                                     trends[cenwave][segment]['lp'],
                                     trends[cenwave][segment]['PID'],
                                     trends[cenwave][segment]['target']),
                                     axis=-1
                                ),
                            hovertemplate=
                                'Rootname: %{customdata[0]}<br>'+
                                'Life_adj: %{customdata[1]}<br>'+
                                'Proposid: %{customdata[2]}<br>'+
                                'Target: %{customdata[3]}'
                                "<extra></extra>"
                        )

                        # add the traces to the plot
                        fig.add_trace(trace, row=1, col=1)
                        fig.add_trace(trace, row=2, col=1)
                    
                    # Add the total wavelength bins used to list
                    tot_wl_bins.append(i+1)

                    # Add the unique label of this mode, regardless of wavelength bin
                    labels.append(f"{trends[cenwave][segment]['grating'][0]}/{segment}/{cenwave}")
            
            # total amount of traces within figure
            ld = len(fig.data)
            
            # This will keep only the wavelength bins of the first mode visible 
            # (etc, first 34 bins * 2 for both plots) 
            for k in range(tot_wl_bins[0]*2, ld):
                fig.update_traces(visible=False, selector=k)
            
            def create_layout_button(k, label):
                # k is number for which element of labels array
                # label is the unique label of this mode

                # Visibility array, fill all False
                # ld -> the number of traces in the figure
                visibility = [False]*ld

                # Sum all previous # of wavelength bins for each mode prior to this specific k value
                tot = 0
                for i in range(k+1):
                    if i != 0:
                        tot+= tot_wl_bins[i-1]*2
    
                # Loop over the visibility array to only select True for ones corresponding to mode selected
                for tr in range(tot, tot_wl_bins[k]*2+tot):
                    visibility[tr] = True

                # dictionary of the button
                return (dict(
                    label=label,
                    method='update',
                    args=[
                        {'visible': visibility,
                         'title':label,
                         'showlegend':True}
                    ]
                ))

            # create buttons
            updatemenus = [
                go.layout.Updatemenu(
                    active=0,
                    buttons=[
                        create_layout_button(k, label) for k, label in enumerate(labels)
                    ]
                ),
            ]

            # add in the menu to the figure
            fig.update_layout(updatemenus=updatemenus)

            # write the plot to the html file
            fig.write_html(f'rel_sens_{name[j]}.html')