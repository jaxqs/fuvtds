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
import datetime
from astropy.convolution import Box1DKernel, convolve
from IPython.display import HTML

"""
This is the base class for the FUVTDS Monitor that will do all
the analysis need and provide the necessary component to the
monitor that will conduct all the plotting.

Please be aware, this will be a monster.

Notes to do in the future:
    - Combine get hdu info with bin data
    - Exchange mpfit.mpfit with scipy.curve_fit
    - Do multithreading to run each cenwave/segment combo in parallel
    - Put outputs into a log file instead out outright outputting
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

    def __init__(self, PIDs, reftime = 54952.0, 
                 breakpoints = [2010.2, 2011.2, 2011.75, 2012.0, 2012.8, 2013.8, 2015.5, 2019.0, 2020.6, 2022.0, 2023.2],
                 inventory='inventory_test.csv'):
        """
        Args:
            PIDs: The dat file that will store all the PIDs part of the
                FUVTDS Monitor programs.
            reftime: The mjd of the reftime.
            breakpoints: All the TDS breakpoints by fractional year.
            inventory: The csv file containing the x1d files and other information used in the FUV TDS Monitor
        """
        self.cenwaves = [1533, 1577, 1623, 1291, 1327, 1222, 1105, 1280, 800, 1055, 1096]
        self.parse_infiles(PIDs, inventory)
        self.breakpoints = np.array(breakpoints)
        self.reftime = Time(reftime, format="mjd").decimalyear
        data_dictionary = self.get_hduinfo(inventory)
        small_dic = self.bin_data(data_dictionary, 'small')
        large_dic = self.bin_data(data_dictionary, 'large')

        # scale between LPs here
        small = self.scale_lps(small_dic)
        large = self.scale_lps(large_dic)

        # scale all to one
        self.small = self.scale_to_1_all_data(small)
        self.large = self.scale_to_1_all_data(large)
            
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
    def scale_lps(self, dictionary):
        """
        Scale the mode between all Lifetime positions present.

        Args:
            dictionary: This dictionary contains the binned flux and date and other information
                        of every monitored mode.
        
        returns:
            dictionary: dictionary where all the monitor modes are scaled between LPs, respectively

        add logging function in this to print out outputs
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
                            #print(f"+++ Scaling LP4 to LP3 using data from datasets: {dictionary[cenwave][segment]['infiles'][lp4_indx[0]-1]} {dictionary[cenwave][segment]['infiles'][lp4_indx[0]]}")

                            # Scale LP3 after LP4 -> LP3
                            if len(lp3_indx2) > 0:
                                dictionary[cenwave][segment]['scale_factor'][lp3_indx2, i] = (dictionary[cenwave][segment]['binned_net'][lp4_indx[0], i] / dictionary[cenwave][segment]['binned_net'][lp4_indx[0]-1, i])
                                dictionary[cenwave][segment]['scaled_net'][lp3_indx2, i] = dictionary[cenwave][segment]['scaled_net'][lp3_indx2, i] * dictionary[cenwave][segment]['scale_factor'][lp3_indx2, i]

                                # calculate error
                                dictionary[cenwave][segment]['scaled_stdev'][lp3_indx2, i] = calc_error(
                                    i, dictionary[cenwave][segment], lp4_indx, lp3_indx2
                                )
                                #print ('+++ Scaling LP3 to LP4 using data from datasets: ', dictionary[cenwave][segment]['infiles'][lp4_indx[0]], dictionary[cenwave][segment]['infiles'][lp4_indx[0]-1])
                        elif cenwave == 800:
                            dictionary[cenwave][segment]['scale_factor'][lp3_indx2, i] = (dictionary[cenwave][segment]['binned_net'][lp3_indx2[0]+1, i] / dictionary[cenwave][segment]['binned_net'][lp3_indx2[0], i])
                            dictionary[cenwave][segment]['scaled_net'][lp3_indx2, i] = dictionary[cenwave][segment]['scaled_net'][lp3_indx2, i] * dictionary[cenwave][segment]['scale_factor'][lp3_indx2, i]
                            # calculate error
                            dictionary[cenwave][segment]['scaled_stdev'][lp3_indx2, i] = calc_error(
                                i, dictionary[cenwave][segment], lp4_indx, lp3_indx2
                            )

                            #print ('+++ Scaling LP3 to LP4 using data from datasets: ', dictionary[cenwave][segment]['infiles'][lp3_indx2[0]+1], dictionary[cenwave][segment]['infiles'][lp3_indx2[0]])

                if (3 in dictionary[cenwave][segment]['lp']) & (2 in dictionary[cenwave][segment]['lp']):

                    lp3_indx_wd308 = np.where((dictionary[cenwave][segment]['lp'] == 3) &
                                            (dictionary[cenwave][segment]['target'] == 'WD0308-565'))
                    lp3_indx_gd71 = np.where((dictionary[cenwave][segment]['lp'] == 3) &
                                            (dictionary[cenwave][segment]['target'] == 'GD71'))
                    
                    lp2_indx_wd308 = np.where((dictionary[cenwave][segment]['lp'] == 2) &
                                            (dictionary[cenwave][segment]['target'] == 'WD0308-565'))
                    lp2_indx_gd71 = np.where((dictionary[cenwave][segment]['lp'] == 2) &
                                            (dictionary[cenwave][segment]['target'] == 'GD71'))
                    
                    if (cenwave > 1500) & (segment == 'FUVA'):
                        lp2_indx = lp2_indx_gd71
                        lp3_indx  = lp3_indx_gd71
                    else: 
                        lp2_indx = lp2_indx_wd308
                        lp3_indx  = lp3_indx_wd308
                    
                    lp2_indx = lp2_indx[0]
                    lp3_indx  = lp3_indx[0]

                    for i, _ in enumerate(dictionary[cenwave][segment]['binned_wl']):
                        # Scale LP3
                        dictionary[cenwave][segment]['scale_factor'][lp3_indx, i] = (
                            dictionary[cenwave][segment]['binned_net'][lp2_indx[-1], i] /
                            dictionary[cenwave][segment]['binned_net'][lp3_indx[0], i]
                        )
                        dictionary[cenwave][segment]['scaled_net'][lp3_indx, i] = (
                            dictionary[cenwave][segment]['scaled_net'][lp3_indx, i] *
                            dictionary[cenwave][segment]['scale_factor'][lp3_indx, i]
                        )
                        
                        # calculate error
                        dictionary[cenwave][segment]['scaled_stdev'][lp3_indx, i] = calc_error(
                            i, dictionary[cenwave][segment], lp2_indx, lp3_indx
                        )
                    print(f"+++ Scaling LP3 to LP2 using data from datasets: {dictionary[cenwave][segment]['infiles'][lp2_indx[-1]]} {dictionary[cenwave][segment]['infiles'][lp3_indx[0]]}")
                

                if (2 in dictionary[cenwave][segment]['lp']) & (1 in dictionary[cenwave][segment]['lp']):
                    lp2_indx_wd308 = np.where(
                        (dictionary[cenwave][segment]['lp'] >= 2) &
                        (dictionary[cenwave][segment]['target'] == 'WD0308-565'))
                    lp2_indx_gd71 = np.where(
                        (dictionary[cenwave][segment]['lp'] >= 2) &
                        (dictionary[cenwave][segment]['target'] == 'GD71'))
                    

                    lp1_indx_wd1057 = np.where(
                        (dictionary[cenwave][segment]['lp'] == 1) &
                        (dictionary[cenwave][segment]['target'] == 'WD1057+719'))
                    lp1_indx_wd0947 = np.where(
                        (dictionary[cenwave][segment]['lp'] == 1) &
                        (dictionary[cenwave][segment]['target'] == 'WD0947+857'))
                    
                    
                    if (cenwave > 1500) & (segment == 'FUVA'):
                        lp1_indx = lp1_indx_wd1057
                        lp2_indx  = lp2_indx_gd71
                    elif (cenwave > 1500) & (segment == 'FUVB'):
                        lp1_indx = lp1_indx_wd1057
                        lp2_indx = lp2_indx_wd308
                    else: 
                        lp1_indx = lp1_indx_wd0947
                        lp2_indx  = lp2_indx_wd308
                    
                    lp1_indx = lp1_indx[0]
                    lp2_indx  = lp2_indx[0]

                    for i, _ in enumerate(dictionary[cenwave][segment]['binned_wl']):
                        # Scale LP2
                        dictionary[cenwave][segment]['scale_factor'][lp2_indx, i] = (
                            dictionary[cenwave][segment]['binned_net'][lp1_indx[-1], i] /
                            dictionary[cenwave][segment]['binned_net'][lp2_indx[0], i]
                        )
                        dictionary[cenwave][segment]['scaled_net'][lp2_indx, i] = (
                            dictionary[cenwave][segment]['scaled_net'][lp2_indx, i] *
                            dictionary[cenwave][segment]['scale_factor'][lp2_indx, i]
                        )
                        
                        # calculate error
                        dictionary[cenwave][segment]['scaled_stdev'][lp2_indx, i] = calc_error(
                            i, dictionary[cenwave][segment], lp1_indx, lp2_indx
                        )
                    print(f"+++ Scaling LP2 to LP1 using data from datasets: {dictionary[cenwave][segment]['infiles'][lp1_indx[-1]]} {dictionary[cenwave][segment]['infiles'][lp2_indx[0]]}")

        return (dictionary)
# --------------------------------------------------------------------------------#
    def bin_data(self, data_dic, size):
        """
        Bin the net counts in each wavelength bin from files that correspond to its
        respective cenwave and segment. The standard deviation is calculated for
        the binning.

        Args:
            data_dic (dictionary): the dictionary containing the net, wavelength, 
                                grating, lp, target, rootname, date, and infiles
                                information for all x1d files of each cenwave
                                and segment setting.
            size (string): Small/Large is used, determines the binsize for each mode
                           and the wavelength edges of each segment. 
        """

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
        
        # Set dictionary that will hold the information used for the FUV TDS Monitor
        # before all the binning and scaling occurs.
        dictionary = {}

        # Only look at the cenwaves used in the monitor
        for cenwave in self.cenwaves:

            # If the cenwave is not in the dictionary, add it.
            if cenwave not in dictionary.keys():
                dictionary[cenwave] = {}

            # Only look at the segments of the cenwaves used in the monitor
            for segment in data_dic[cenwave]:

                # If the segment is not in the dictionary[cenwave], add it.
                if segment not in dictionary[cenwave].keys():

                    # Declare the keys of the dictionary[cenwave][segment]
                    dictionary[cenwave][segment] = {
                        'binned_net': [],
                        'binned_wl' : [],
                        'wl_bin_edges': [],
                        'stdev': [],
                        'grating': [],
                        'lp': [],
                        'target': [],
                        'rootname': [],
                        'date': [],
                        'infiles': [],
                        'PID': []
                    }
                
                # wl_range contains minimun wavelength, maximun wavelength, and binsize
                wl_range = wl_info_dict[size][cenwave][segment]
                min_wl = wl_range[0]
                max_wl = wl_range[1]
                binsize = wl_range[2]

                # the wavelength bin edges based on binsize
                bins = np.arange(min_wl, max_wl+1, binsize)

                # look at each wavelength array of each x1d file
                for i, wl in enumerate(data_dic[cenwave][segment]['wavelength']):

                    # the wavelength values that fall within the wavelength range of the segment
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

                # Take the information from the data_dic into the dictionary for binned data
                dictionary[cenwave][segment]['binned_wl'] = np.array(edges[:-1]+np.diff(edges)/2)
                dictionary[cenwave][segment]['wl_bin_edges'] = np.array(edges)
                dictionary[cenwave][segment]['best_fit'] = np.empty((
                    len(dictionary[cenwave][segment]['binned_wl']), (len(self.breakpoints)+1)*2
                    ))
                dictionary[cenwave][segment]['best_fit_err'] = np.empty((
                    len(dictionary[cenwave][segment]['binned_wl']), (len(self.breakpoints)+1)*2
                    ))
                dictionary[cenwave][segment]['grating'] = data_dic[cenwave][segment]['grating']
                dictionary[cenwave][segment]['lp'] = data_dic[cenwave][segment]['lp']
                dictionary[cenwave][segment]['target'] = data_dic[cenwave][segment]['target']
                dictionary[cenwave][segment]['rootname'] = data_dic[cenwave][segment]['rootname']
                dictionary[cenwave][segment]['date'] = data_dic[cenwave][segment]['date']
                dictionary[cenwave][segment]['infiles'] = data_dic[cenwave][segment]['infiles']
                dictionary[cenwave][segment]['PID'] = data_dic[cenwave][segment]['PID']

        # reformat + add scaled components to dictionary
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

                # best fit and best fit error
                dictionary[cenwave][segment]['best_fit']     = np.empty( (len(dictionary[cenwave][segment]['binned_wl']), (len(self.breakpoints) + 1)*2))
                dictionary[cenwave][segment]['best_fit_err'] = np.empty( (len(dictionary[cenwave][segment]['binned_wl']), (len(self.breakpoints) + 1)*2))
        
        # Don't save it as a class component yet because all the math hasn't been done yet.
        return (dictionary)

# --------------------------------------------------------------------------------#
    def get_hduinfo(self, csv_file):
        """
        Obtain the header and data information for all the x1d files in the programs used
        in the FUV TDS Monitor. Only x1d file information is taken for the currently monitored
        cenwaves.

        A csv file that contains the header information is used to obtain the file path
        in order to reduce the time it takes to run through all the x1d files and instead
        only look at files that align with that cenwave, for each cenwave.

        Args:
            csv_file: The csv file that contains the header information and file path of the
            x1d files used in this run of the monitor.
        
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

        # read in inventory file as a pandas DataFrame
        inventory = pd.read_csv(csv_file)

        # the dictionary that will hold the data informaton from the x1d files
        data_dic = {}

        for cenwave in self.cenwaves:

            # if cenwave is not in the dictionary, add it
            if cenwave not in data_dic.keys():
                data_dic[cenwave] = {}
            
            # Obtain array-like list of only the x1d files of this cenwave from dataframe
            files = np.array(inventory['file_path'][(inventory['cenwave'] == cenwave)]).flatten()
            
            # iterate over the x1d files of the cenwave
            for file in files:
                with fits.open(file, memmap=False) as hdulist:
                    hdr0 = hdulist[0].header
                    hdr1 = hdulist[1].header
                    data = hdulist[1].data

                    # iterate over the segments of the x1d file. If there is only one segment
                    # used, this will only iterate once. If two segments are used, then this
                    # will iterate twice. data['segment'] will list all segments used. Also
                    # applies to NUV data as well.
                    for i, segment in enumerate(data['segment']):

                        # This line of code removes targets no longer used for that specific mode in TDS
                        if hdr0['targname'] not in targ_info_dict[cenwave][segment]:
                            continue

                        # if segment is not in the dictionary, add it
                        if segment not in data_dic[cenwave].keys():
                            data_dic[cenwave][segment] = {
                                'net': [],
                                'wavelength': [],
                                'grating': [],
                                'lp': [],
                                'target': [],
                                'rootname': [],
                                'date': [],
                                'infiles': [],
                                'PID': []
                            }

                        # Add the data and header information into the data dictionary to be used later
                        data_dic[cenwave][segment]['net'].append(np.array(data['net'][i][data['dq_wgt'][i] != 0]))
                        data_dic[cenwave][segment]['wavelength'].append(np.array(data['wavelength'][i][data['dq_wgt'][i] != 0]))

                        data_dic[cenwave][segment]['grating'].append(hdr0['opt_elem'])
                        data_dic[cenwave][segment]['lp'].append(hdr0['life_adj'])
                        data_dic[cenwave][segment]['target'].append(hdr0['targname'])
                        data_dic[cenwave][segment]['rootname'].append(hdr0['rootname'])
                        data_dic[cenwave][segment]['date'].append(Time(hdr1['date-obs'], format='fits').decimalyear) # change from mjd to decimal year
                        data_dic[cenwave][segment]['infiles'].append(file)
                        data_dic[cenwave][segment]['PID'].append(hdr0['proposid'])

        # change to np.array
        for cenwave in data_dic:
            for segment in data_dic[cenwave]:
                data_dic[cenwave][segment]['grating'] = np.array(data_dic[cenwave][segment]['grating'])
                data_dic[cenwave][segment]['lp'] = np.array(data_dic[cenwave][segment]['lp'])
                data_dic[cenwave][segment]['target'] = np.array(data_dic[cenwave][segment]['target'])
                data_dic[cenwave][segment]['rootname'] = np.array(data_dic[cenwave][segment]['rootname'])
                data_dic[cenwave][segment]['date'] = np.array(data_dic[cenwave][segment]['date'])
                data_dic[cenwave][segment]['infiles'] = np.array(data_dic[cenwave][segment]['infiles'])
                data_dic[cenwave][segment]['PID'] = np.array(data_dic[cenwave][segment]['PID'])
        
        # Don't save it as a class component yet because all the math hasn't been done yet.
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


class FUVTDSMonitor(object):
    """
    This plotting function
    """

    def __init__(self, TDSData) -> None:
        """
        explaining here
        """
        self.trends = TDSData
        self.solar  = self.get_solar_data()

    def get_solar_data(self):
        """
        explain here
        """
        spaceweather_url = 'ftp://ftp.seismo.nrcan.gc.ca/spaceweather/solar_flux/daily_flux_values/fluxtable.txt'

        request.urlretrieve(spaceweather_url, 'fluxtable.txt')

        data = ascii.read('fluxtable.txt')
        df = data.to_pandas()
        fluxjulian = df['fluxjulian']
        df.index = pd.DatetimeIndex(fluxjulian)
        df['date'] = Time(fluxjulian, format='jd').decimalyear

        df = df.drop(columns=['fluxdate', 'fluxtime', 'fluxcarrington', 'fluxadjflux', 'fluxursi'])
        df.rename(columns = {'fluxobsflux':'f10.7'}, inplace = True)
        df = df.reindex(columns=['date', 'f10.7'])

        # this removes any previous solar flux txt that is out of date
        os.system('rm '+os.path.join("solar_flux.txt"))

        outfile = os.path.join("solar_flux.txt")
        df = df[df['date'] >= 2009.5]
        df.to_csv(outfile, header=None, index=None, sep=' ', mode='a')

        return(df)
    
    def plot_solar_flux(self):
        """smth"""

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
        for i, cenwave in enumerate(self.trends.large):
            for j, segment in enumerate(self.trends.large[cenwave]):
                data = go.Scatter(
                    x=self.trends.large[cenwave][segment]['date'],
                    y=self.trends.large[cenwave][segment]['scaled_net'][:,0],
                    mode='markers',
                    name=f'{cenwave}/{segment}',
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
                    "<extra></extra>",

                )
                fig.add_trace(data, secondary_y=False)
        
        fig.add_trace(unsmoothed, secondary_y=True)
        fig.add_trace(smoothed, secondary_y=True)

        fig.update_layout(
            title_text="TDS Solar Flux"
        )

        # set x-axis title
        fig.update_xaxes(title_text="Date")
        
        # set y-axes titles
        fig.update_yaxes(title_text="Fractional Throughput", range=(0.0, 1.1), secondary_y=False)
        fig.update_yaxes(title_text="10.7 cm Flux (units here)", range=(50, 400), secondary_y=True)
        fig.show()
        fig.write_html('tds_solar_flux.html')

