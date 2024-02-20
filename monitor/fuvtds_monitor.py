import os
import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
import plotly.graph_objs as go

from peewee import Model
from typing import Any, Union
from urllib import request
from itertools import repeat
from plotly.subplots import make_subplots
from monitorframe.monitor import BaseMonitor
from astropy.convolution import Box1DKernel, convolve
from astropy.time import Time
from astropy.io import ascii
from scipy.stats import tmean, tstd

from data_models import FUVTDSModel
from monitor_helpers import fit_line, convert_day_of_year

COS_MONITORING = 'output/'
spaceweather_url = 'ftp://ftp.seismo.nrcan.gc.ca/spaceweather/solar_flux/daily_flux_values/fluxtable.txt'

def binned(wavelength_x, net_y, dq_wgt, cenwave, segment):
    """
    This function bins the wavelength (x array) and the net (y array) by pre-established
    binwidth that will aid with the fitting process. 
    """
    # establish the wavelength bins
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

    wl_bin_edges = np.arange(
        wl_info_dict[cenwave][segment][0], #wl min
        wl_info_dict[cenwave][segment][1], #wl max
        wl_info_dict[cenwave][segment][2]  #binwidth
        )
    wl_bin_edges = np.append(wl_bin_edges, wl_info_dict[cenwave][segment][1])

    wavelength = []
    net = []
    stdev = []
    for edges in range(len(wl_bin_edges))[0:-1]:
        index = np.where((wavelength_x >= wl_bin_edges[edges]) & 
                         (wavelength_x < wl_bin_edges[edges+1]) &
                         (dq_wgt != 0.0))
        wavelength.append(tmean(wavelength_x[index]))
        net.append(tmean(net_y[index]))
        stdev.append(tstd(net[index])/np.sqrt(len(net[index])))
    wavelength = np.array(wavelength)
    net = np.array(net)
    stdev = np.array(stdev)

    return(wavelength, net, stdev)

def select_sci(model: Union[Model, None], exptype: str, target: str, new_data_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Get all ingested fuvtds data of a particular exptype and target and combine it with any new data found.
    """
    data = pd.DataFrame()

    if model is not None:
        data = data.append(
            pd.DataFrame(model.select().where((model.EXPTYPE == exptype) & (model.TARGNAME != target)).dicts()),
            sort=True,
            ignore_index=True
        )
    
    if new_data_df is None:
        return data

    if not new_data_df.empty:
        new_data = new_data_df[(new_data_df.EXPTYPE == exptype) &(new_data_df.TARGNAME != target)].reset_index(drop=True)
        data = data.append(new_data, sort=True, ignore_index=True)
    
    return (data)

def get_solar_data() -> pd.DataFrame:
    """
    Download the most recent solar data, save as file and dataframe,
    filter dataframe to date range. also replace -1 values in the smoothed flux
    """
    request.urlretrieve(spaceweather_url, COS_MONITORING+'fluxtable.txt')

    data = ascii.read(COS_MONITORING+'fluxtable.txt')
    df = data.to_pandas()
    fluxjulian = df['fluxjulian']
    df.index = pd.DatetimeIndex(fluxjulian)

    t = Time(fluxjulian, format='jd')
    timedecimal = t.decimalyear
    df['date'] = timedecimal

    df = df.drop(columns=['fluxdate', 'fluxtime', 'fluxcarrington', 'fluxadjflux', 'fluxursi'])
    df.rename(columns={'fluxobsflux': 'f10.7'}, inplace=True)
    df = df.reindex(columns=['date', 'f10.7'])

    outfile = os.path.join(COS_MONITORING, 'solar_flux.txt')
    df.to_csv(outfile, header=None, index=None, sep=' ', mode='a')

    return (df)

class FUVTDSMonitor(BaseMonitor):
    """
    A skeleton of the FUVTDS Monitor
    """
    data_model = FUVTDSModel
    output = COS_MONITORING
    docs = "insert link here LOL!"
    labels = ['ROOTNAME', 'LIFE_ADJ', 'PROPOSID', 'OPT_ELEM', 'CENWAVE', 'SEGMENT'] # labels of specific datapoints

    subplots = True
    subplot_layout = (2, 1) # 2 rows, 1 column

    run = 'monthly'

    # Define TDS breakpoints in year.days -- currently in fraction years!!!!
    break_points = {
        'TDS': [
            (None, 2010.2), 
            (2010.2, 2011.2),
            (2011.2, 2011.75),
            (2011.75, 2012.8), 
            (2012.8, 2013.8), 
            (2013.8, 2015.5), 
            (2015.5, 2019.0), 
            (2019.0, 2020.6),
            (2020.6, 2022.0),
            (2022.0, None)
        ]
        }

    # Define import events for vertical line placement
    # Format is -> Event: year.days since jan1st ; right now fractional year!
    # this will have HV raises, LP switches, Voltage change for A and B, and TDS breakpoints
    events = {
        'TDS Breakpoint 1': 2010.2,
        'Voltage B 1': 2011.18,
        'TDS Breakpoint 2': 2011.2,
        'TDS Breakpoint 3': 2011.75,
        'Voltage A 1': 2012.23,
        'Move to LP2': 2012.12,
        'Voltage A 2': 2012.56,
        'Voltage B 2': 2012.56,
        'TDS Breakpoint 4': 2012.8,
        'Voltage B 3': 2013.47,
        'TDS Breakpoint 5': 2013.8,
        'Voltage B 4': 2014.55,
        'Voltage A 3': 2014.84,
        'TDS Breakpoint 6': 2015.5,
        'Move to LP3': 2015.107,
        'Voltage A 4': 2015.107,
        'Voltage B 4': 2015.107,
        'Voltage B 5': 2016.05,
        'Move to LP4': 2017.75,
        'Voltage A 5': 2017.75,
        'Voltage B 6': 2017.75,
        'TDS Breakpoint 7': 2019.0,
        'TDS Breakpoint 8': 2020.6,
        'Voltage A 6': 2020.75,
        'Voltage B 7': 2020.75,
        'Move to LP5': 2021.76,
        'TDS Breakpoint 9': 2022.0,
        'Voltage B 8': 2022.47,
    } 

    def get_data(self):
        """
        Filter EXTERNAL/SCI and certain target data for the FUVTDS plot. This filter attempt will weed out the
        WAVECAL files and targets that are no longer used in the FUVTDS analysis.
        """
        data = select_sci(self.model.model, 'EXTERNAL/SCI', 'LDS749B', self.model.new_data)
        # another function to change the x and y stuff? like in dark_filter in the dark monitor?
        try:
            data['BINNED_WL'], data['BINNED_NET'], data['BINNED_STDEV'] = binned(data.WAVELENGTH, data.NET, data.DQ_WGT, data.CENWAVE, data.SEGMENT)
        except:
            pass #for when the segment is BOTH, need to get a work around

        filtered_df = data[
            (data.EXPTYPE == 'EXTERNAL/SCI') &
            (data.TARGNAME != 'LDS749B') &
            (data.EXPSTART > 56130.0) &
            (data.CENWAVE != 1280) &
            (data.SEGMENT == 'FUVA')
        ]

        return (filtered_df.sort_values('EXPSTART').reset_index(drop=True))
    
    def track(self):
        """
        Track the fit and fit line for the period since the last TDS breakpoint.
        """
        last_updated_results = {}
        
        t_start = convert_day_of_year(self.break_points['TDS'][-1][0]).mjd # Last TDS Breakpoint

        df = self.data[self.data.EXPSTART >= t_start]

        if df.empty:
            pass
        tds_fit, tds_line = fit_line(Time(df.EXPSTART, format='mjd').byear, df.BINNED_NET)

        last_updated_results = {
            'slope': tds_fit[1], 'start': tds_line[0], 'end': tds_line[-1]
        }
        return (last_updated_results)
