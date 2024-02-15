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

from data_models import FUVTDSModel
from monitor_helpers import explode_df, absolute_time

COS_MONITORING = 'output/'
spaceweather_url = 'ftp://ftp.seismo.nrcan.gc.ca/spaceweather/solar_flux/daily_flux_values/fluxtable.txt'

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

def get_solar_data():
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

    # Define TDS breakpoints in year.days
    break_points = {}

    # Define import events for vertical line placement
    # Format is -> Event: year.days since jan1st
    events = {} # this will have HV raises, LP switches, Voltage change for A and B, and TDS breakpoints

    def get_data(self):
        """
        Filter EXTERNAL/SCI and certain target data for the FUVTDS plot. This filter attempt will weed out the
        WAVECAL files and targets that are no longer used in the FUVTDS analysis.
        """
        data = select_sci(self.model.model, 'EXTERNAL/SCI', 'LDS749B', self.model.new_data)

        # another function to change the x and y stuff? like in dark_filter in the dark monitor?
