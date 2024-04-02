import numpy as np
####
def compute_chi2(y, err, model, npars):
    """Compute chi-squared and degrees of freedom for a fit."""
    term = (y - model) / err
    chi2 = np.sum(term**2)
    dof = len(y) - npars
    return chi2, dof

def broken_lines(x, *a):
    """
    Fitting function for a segmented line with n_bp breakpoints.
    
    Args:
        x: date - reftime
        *a: initial parameters guess (p0, p)
    """

    # y values are zeros, initial to then replace. same length as amount of date values / exposures
    y = np.zeros(len(x))

    # number of breakpoints
    n_bp = 0

    # number of parameters 
    n_pars = len(a)

    if n_pars == 1:

        #idk lmao
        y = [a[0] for e in x]

    elif n_pars == 2:

        # if there are only two parameters (aka zero breakpoints), then do a single linear fit
        # for each time value of the exposures (date - reftime)
        y = [a[0] + a[1] * e for e in x]

    # If there is an even amount of parameters, do this
    elif not len(a) % 2:


        # removes the intercept and slope (first two in a [p0, initial parameters]) and counts
        # the number of breakpoints we are working with
        n_bp = (len(a) - 2) // 2

        # x-values == number of breakpoints, time-related
        x_bp = np.zeros(n_bp)

        # y-values == number of breakpoints, scaled net count related
        y_bp = np.zeros(n_bp)

        # set first breakpoint to first value of x-array, time-related
        x_bp[0] = a[2]

        # do linear fit where
        #       a[0]: intercept, intial parameters
        #       a[1]: slope, initial parameters
        #       a[2]: first breakpoint, time-related
        y_bp[0] = a[0] + a[1] * a[2]

        # Loop over the amount of breakpoints, skipping the first breakpoint as it was already used above
        for j in range(1, n_bp):

            # set x value to the next breakpoint
            x_bp[j] = a[2 * (j + 1)]

            # set the y value where the last yvalue is the intercept 
            # added to the next breakpoint available that is then multiplied..
            #       y_bp[j - 1]: last y value, intercept
            #       a[2 * j + 1]: the next slope value of this breakpoint, slope
            #       x_bp[j] - x_bp[j - 1]: subtract the last breakpoint to the current one to be time-related, time-related x value
            y_bp[j] = y_bp[j - 1] + a[2 * j + 1] * (x_bp[j] - x_bp[j - 1])
        
        # loop over the number of xvalues, aka number of date values (all exposures)
        for i in range(len(x)):

            # get first in time line segment
            if x[i] < x_bp[0]:  # The first line segment.

                # do a linear fit with
                #       a[0]: intercept, initial parameters
                #       a[1]: slope, initial parameters
                #       x[i]: first date in time, time-related
                yy = a[0] + a[1] * x[i]
            
            # get the last in time line segment
            elif x[i] >= x_bp[-1]:  # The last line segment.

                # do a lienar fit with
                #       y_bp[-1]: last yvalue breakpoint related, intercept
                #       a[-1]: last slope value, initial parameters
                #       (x[i] - a[-2]): last date in time subtracted by last breakpoint in initial parameters
                yy = y_bp[-1] + a[-1] * (x[i] - a[-2])
            
            # get the in between line segments
            elif n_bp > 1:  # All the little line segments in between.

                # loop over the breakpoints
                for j in range(n_bp - 1):

                    # get the date values inbetween the breakpoints
                    if x[i] >= x_bp[j] and x[i] < x_bp[j + 1]:

                        # do a linear fit with
                        #       y_bp[j]: yvalue of this time period as intercept
                        #       a[2 * (j + 1) + 1]: slope value corresponding to this time segment, slope
                        #       (x[i] - x_bp[j]): date in time subtracted by this breakpoint, time-related
                        yy = y_bp[j] + a[2 * (j + 1) + 1] * (x[i] - x_bp[j])
                        break
            
            # put the fitted yvalues into an array
            y[i] = yy
    else:
        print('warning, number of fit parameters ({0})'.format(len(a)), end='')
        print('is not even, fit may be rubbish')
    return y

def mpfitting_function(p, fjac=None, x=None, y=None, err=None):
    """
    args:
        p: p0, initial parameters
        fjac: partial derivatives?
        x: date - reftime
        y: scaled net count rate 
        err: scaled stdev of scaled net count rate
    """

    """Mpfit fitting function."""
    status = 0
    if fjac is not None:
        print('mpfitting_function does not calculate partial derivatives')
        status = -1
    if x is None:
        print('mpfitting_function requires that x be provided')
        status = -2
    if y is None:
        print('mpfitting_function requires that y be provided')
        status = -2
    if err is None:
        print('the error vector is assumed to be unity')
        err = np.ones(len(x))
    elif 0.0 in err[:]:
        print('zero value in error vector')
        status = -3
    if status == 0:

        # the actual function that is done
        # uses date-reftime and intial parameters (p0 or p)
        # returns the y-values of all the linear fits
        model = broken_lines(x, *p)
        
        # subtract the model y-values from the linear fits (model) from the 
        # observed y-values (scaled net count) and divide by stdev error
        weighted_deviations = (y - model) / err
    return [status, weighted_deviations]

def wmean(x, w):
    """Return the weighted mean, unbiased weighted sample variance,
    and the weighted standard error."""

    # See
    # <http://en.wikipedia.org/wiki/Mean_square_weighted_deviation> on
    # 2012-06-15 at 1415 EDT for details about the weighted mean and
    # unbiased weighted variance.

    if len(x) <= 0:  # There is no data.
        return None, None, None
    elif len(x) == 1:  # No variance or standard error
        return x[0], 0.0, 0.0

    # Weighted mean.
    wmeanx, wsum = np.ma.average(x, weights=w, returned=True)

    # Unbiased weighted sample variance
    wsum2 = wsum**2
    w2sum = np.sum(w**2)
    n = len(w)
    if abs(wsum2 - w2sum) < np.finfo(np.dtype(wsum2)).eps:
        s2 = 1.0 / (n - 1) * np.sum((x - wmeanx)**2)
    else:
        s2 = (np.sum(w * x**2) * wsum - np.sum(w * x)**2) / (wsum2 - w2sum)

    # Standard error in the weighted mean.
    # See Gatz & Smith (2007, Atmospheric Environment, 11, p1185) and
    # <http://stats.stackexchange.com/questions/25895/computing-standard-error-in-weighted-mean-estimation>
    # on 2012-11-06 at 0900 EDT.
    t1 = n / ((n - 1) * wsum2)
    meanw = np.mean(w)
    t2 = np.sum((w * x - meanw * wmeanx)**2)
    t3 = wmeanx * np.sum((w - meanw) * (w * x - meanw * wmeanx))
    t4 = wmeanx**2 * np.sum((w - meanw)**2)
    wse = np.sqrt(t1 * (t2 - 2.0 * t3 + t4))

    return wmeanx, s2, wse


def mpfit_the_data(func, x, y, err=None, p0=None, parinfo=None):
    """
    Args:
        func (function): mpfit function
        x (array): date - reftime
        y (array): scaled net count rate 
        err (array): stdev of scaled net count rate
        p0: initial parmeters guess
        parinfo: list of dictionaries with parameter information
    """

    """Call the fitting function.  Uses mpfit."""
    if err is None:
        err = np.ones(len(x))
    if p0 is None:
        print('error, initial parameter guesses not provided')
        return None, None, None
    if len(p0) == 1:  # Zero line segments.
        w = 1.0 / err**2
        popt, cov, perr = wmean(y, w)
        return [popt], [cov], [perr]
    fa = {'x':x, 'y':y, 'err':err}

    # conduct mpfit
    #   fcn: mpfitting_function; the called function
    #   xall: p0; initial parameter guess
    #   functkw: fa; dictionary which contains the paraeters to be passed in the user-supplied function
    #                   aka, the x, y, err values (date, scaled net count, stdev error)
    #   parinfo: parinfo; mechanism for more sophisticated constraints to be placed on parameter values
    #                       value: the starting parameter value (breakpoint - reftime)
    #                       fixed: boolean, if the parameter is to be held fixed or not
    #                       limited: two element boolean array. If set, then parameter is bounded on the lower/upper side
    #                       limits: two element float array. gives parameter limits on lower and upper ALL N/A FOR US
    #                       step: the step size to be used in calculating numerical deviations
    #                       mpside: the sidedness of the finite differences when computing 
    #                               0 means one-sided derivative computed automatically
    #                       mpmaxstep: maximum change to be made in parameter value (none)
    #                       tied: string expression that 'ties' the parameter to other free or
    #                            fixed parameters. 
    #                       mpprint: if set to 1, default iterfunct will print the parameter value
    #   quiet: True; boolean, if textual output should be printed
    #   debug: 
    #
    # outputs
    #   m, that contains parameters, covarience, and parameter errors
    m = mpfit.mpfit(mpfitting_function, p0, functkw=fa, parinfo=parinfo,
                    quiet=True, debug=False)

    if m.status <= 0:
        print('mpfit, status={0}, {1}'.format(m.status, m.errmsg))
        popt, pcov, perr = None, None, None

    # grab the parameters, covarience, and parameters errors    
    popt = m.params
    pcov = m.covar
    perr = m.perror
    return popt, pcov, perr


def create_parinfo_list(p0,x,blue_flag):
    '''
    Create the parinfo list of dictionaries for mpfit

    Parameters
    -----------
    p0 : list
        a list of best guess input parameters for fit_tds_2.mpfit_the_data

    x : array
        an array of dates of exposures minus reftime
    blue_flag: boolean
        If the cenwave in particular is a bluemode

    Returns:
    -----------
    parinfo : list
        a dictionary of input parameter behavior to pass to fit_tds_2.mpfit_the_data.
        This is used to fix the breakpoint values (not try to find the best fit)
    '''

    # Find the first segment with data

    # create array the length of initial parameters list
    flag = np.zeros(len(p0),dtype=int)

    # create array the length of all breakpoints, will contain all breakpoints (~11, currently there are 12 bp)
    bp = np.zeros((len(p0)-2)//2)

    # set last flag as one to be an edge/boundary
    flag[-1]=1

    # look over length of initial parameters list starting at 2 for 2-stepsize
    # OR loop over the amount of breakpoints
    for j in range(2,len(p0),2):

        # set intercept value from initial parameters into its own array (bp)
        bp[(j-2)//2] = p0[j]

        # Set flag index = 1 if the breakpoint in initial_parameters is greater or equal to first in time date (x)
        if p0[j] >= x[0]:
            flag[j]=1 # Flag index corresponding the breakpoint index in initial parameters
            flag[j-1]=1 # Flag index before breakpoint index in initial parameters
            flag[j+1]=1 # Flag index after breakpoint index in initial parameters

    # get the index of the first slope that will be fitted
    idx = np.where(flag == 1)[0][0]  # index of the first slope to be fitting
    tied = 'p['+str(idx)+']' # This can look like p[3] for example

    #
    # bd = [ 0.87123288  1.87123288  2.42123288  2.67123288  3.47123288  4.47123288
    #     6.17123288  9.67123288 11.27123288 12.67123288 13.87123288] DATE!!! DATE - REFTIME


#    pdb.set_trace()
# Do not fit slope of first time segment if less than 4 points for blue mode
    # Blue mode check
    # if first in time date (x) is less than last breakpoint in time and a blue flag, do this
    if x[0] < bp[-1] and blue_flag == True:
        # If there are less than 4 points for blue mode in first in time (x < bp[(idx-1)//2])
        # then set flag in those indx to be -1 (to be excluded) and look at next two indx
        if ( len(np.where(x < bp[(idx-1)//2])[0]) < 4) :
            flag[idx] = -1
            tied = 'p['+str(idx+2)+']'
#    pdb.set_trace()



    # list that will hold all the dictionaries
    parinfo = []

    # loop over length of initial paramters list
    for j in range(len(p0)):

        # 
        if (j > 1) and (j%2 == 0):  #Fix the breakpoints
            parinfo.append({
                'value':p0[j], #breakpoint - reftime
                'fixed':True, 
                'limited':[False, False], 
                'limits':[0, 0], 
                'step': None, 
                'mpside':0, 
                'mpmaxstep':0, 
                'tied':'', 
                'mpprint':1})
        else:
            if (j > 0) and (flag[j] == 0):
                # Fix slopes of segment with no data to slope of first segment with data 
                # (in practice, assumes all segment have data if breakpoint is after the 1st data
                p0[j]=0.0
                parinfo.append({'value':p0[j], 
                                'fixed':True, 
                                'limited':[False, False], 
                                'limits':[0, 0], 
                                'step': None, 
                                'mpside':0, 
                                'mpmaxstep':0, 
                                'tied':'', 
                                'mpprint':1})
            elif (flag[j] == -1):
                # For blue modes, ties slope of 1st segment with data to slope of the 2nd segment if 
                # the 1st segment has less than 4 measurements (fit results were unstable from one wavelength bin to another)
                parinfo.append({'value':p0[idx], 
                                'fixed':True, 
                                'limited':[False, False], 
                                'limits':[0, 0], 
                                'step': None, 
                                'mpside':0, 
                                'mpmaxstep':0, 
                                'tied':tied, 
                                'mpprint':1})
            else:
                parinfo.append({'value':p0[j], 
                                'fixed':False, 
                                'limited':[False, False], 
                                'limits':[0, 0], 
                                'step': None, 
                                'mpside':0, 
                                'mpmaxstep':0, 
                                'tied':'', 
                                'mpprint':1})
#    pdb.set_trace()
    return parinfo

def create_parameter_guess(self, fit_date, fit_net, breakpoints):
    '''
    Use a linear polynomial fit to the data to create a list of the i
    initial parameters for the continuous piecewise function fitting

    Parameters
    ----------
    fit_date : numpy array (x)
        date array to use as independent variable in linear fit
    fit_net : numpy array (y)
        net count array to use as dependent variable in linear fit
    breakpoints : list
        list of dates where linear trend changes

    Returns
    ---------
    initial_params : list
        list of best guess input parameters for mpfit
    '''
    #Don't use self for breakpoints, fit_new, and fit_date to make it easier to use
    #when we want discontinuous breakpoints

    #initial_params = [1, 0] #first zeropoint and slope
    # List of initial parameters for linear fit
    initial_params = list(np.polyfit(fit_date, fit_net, 1))

    #polyfit returns the highest order coefficient first
    # Begin with intercept then slope [intercept, slope]
    initial_params.reverse()

    # loop for amount of breakpoints
    for bp in breakpoints:
        #append breakpoint minus reftime to end of initial parameters list
        initial_params.append(bp )   #breakpoints date minus ref_time

        # append slope of that breakpoint (minus ref time) to end of initial paramters list
        initial_params.append(initial_params[1])   #start with each slope identical
    
    # order goes something like this
    # [intercept, slope1, slope 2, breakpoint1, slope1, breakpoint2, slope2..]
    return initial_params


def fit_the_data(self, dependent_array, dependent_array_error):
    '''
    Perform a piece wise linear fit to the data provided in self.date and self.scaled_net
    Breakpoints are currently held fixed, fit is done using mpfit
    Parameters
    ----------
    dependent_array : numpy array
        array of dependent values (independent variable is self.date) to fit
    dependent_array_error : numpy array
        array of errors of the dependent_array

    Returns
    --------
    fit : numpy array
        array of parameters which define a piecewise linear fit.
        Format is [y_intercept @ reftime, slope1, breakpoint1 - reftime, slope2, breakpointp2 - reftime...]

    '''
    if self.date is None:
        raise NameError('self.date must be defined before this function can be called')

    # Breakpoints from time 0.0 (0.0 being reftime)
    bp = self.breakpoints - self.reftime

    # Date of exposures from time 0.0 (0.0 being reftime)
    x = self.date - self.reftime

    # Scaled net of given cenwave and segment; scaled_net[:, i].ravel()
    y = dependent_array

    # Scaled stdev of given cenwave and segment; scaled_stdev[:, i].ravel()
    error = dependent_array_error

    # remove NaN values from y-array (scaled_net[:, i].ravel()) and from corresponding x-array and error
    x = x[~np.isnan(y)]
    error = error[~np.isnan(y)]
    y = y[~np.isnan(y)]

    # initial parameters = [intercept, slope, breakpoint1, slope of breakpoint1, breakpoint2, slope of breakpoint2]
    initial_params = create_parameter_guess(x, y, bp) # uses date of exposures from reftime, scaled net, and breakpoints

    # Boolean check if this cenwave is a bluemode cenwave
    if (self.cenwave == 1222) or (self.cenwave == 1096) or (self.cenwave == 1055):
        blue_flag = True
    else:
        blue_flag = False

    # parinfo = list of dictionaries that will be used in the mp fitting function
    parinfo = create_parinfo_list(initial_params, x, blue_flag)

#        pdb.set_trace()
    # conduct the fit here
    fit = mpfit_the_data(mpfitting_function, x, y, err = error, p0 = initial_params, parinfo = parinfo)
#        pdb.set_trace()


    # The fit values 
    pars = fit[0]
    cov = fit[1]
    perrs = fit[2]

    # THIS IS NOT USED ANYUMORE. CAN OMIT vvvvvvvvvv
    model = broken_lines(x, *pars)
    chi2, dof = compute_chi2(y, error, model, len(pars))
    chi2nu = chi2 / dof
    residual = model - y
    ndata = len(x)
    rms = np.sqrt(np.sum(residual**2) / ndata)
    p = 1.0 - float(scipy.special.chdtrc(dof, chi2))
    n_seg = len(pars)/2
    # THIS IS NOT USED ANYUMORE. CAN OMIT ^^^^^^^^^^^

    result = [pars, cov, perrs, rms, chi2, dof, p, n_seg, model]

    return (result)

def scale_to_1_all_data(self):
    '''
    Scale all of the scaled_net data so that the fit to the relative net counts is 1 at self.reftime

    Returns
    ------------
    None, modifies: stdev, scaled_net, best_fit, best_fit_error

    '''

    if self.stdev is None:
        raise NameError('Must define self.stdev before calling this method')
    for i in range(len(self.wl_bin_edges) - 1):

        #pdb.set_trace()

        fit = fit_the_data(self.scaled_net[:, i].ravel(), self.scaled_stdev[:, i].ravel())

        # used in the linear fit
        self.best_fit[i, :] = fit[0] # pars
        self.best_fit_error[i, :] = fit[2] # perrs

        if fit[0] is None:
            print ('fit[0] is None')
            pdb.set_trace()
        else:

            # this can be omitted vvvvvv
            self.best_fit_rms[i] = fit[3]
            self.best_fit_chi2[i] = fit[4]
            self.best_fit_dof[i] = fit[5]
            # this can be omitted ^^^^^^^^^^


            y_intercept = self.best_fit[i, 0]
            y_inter_stdev = self.best_fit_error[i, 0]

            # scale to 1
            self.scaled_stdev[:, i] =  self.scaled_stdev[:, i]/y_intercept
            self.scaled_net[:, i] = self.scaled_net[:, i]/y_intercept
            self.best_fit[i, 1::2] = self.best_fit[i, 1::2] / y_intercept
            self.best_fit_error[i, 1::2] = prop_error_for_div(self.best_fit[i, 1::2], self.best_fit_error[i, 1::2], y_intercept, y_inter_stdev)
            
            # this can be omitted vvvvvv
            self.best_fit_rms[i] = self.best_fit_rms[i]/y_intercept
            self.best_fit_chi2[i] = self.best_fit_chi2[i]/y_intercept/y_intercept
            # this can be omitted ^^^^^^^^^^

            # use this still idk lol
            self.best_fit[i, 0] = 1.0