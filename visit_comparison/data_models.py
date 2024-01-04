"""
    IMPORTANT TO NOTE:
        This file was taken from the monitors file 'data_models.py' written by [author].

        The following code has been heavily edited and repurposed for the development 
        of the FUVTDS monitor by Jaq Hernandez.
"""

import pandas as pd
import os

from filesystem import find_files, data_from_exposures

FILES_SOURCE = '/grp/hst/cos2/cosmo'

def get_new_data(PID, visit=''):
    header_request = {
            0: ['ROOTNAME', 'SEGMENT', 'CENWAVE', 'TARGNAME', 'OPT_ELEM', 'DATE']
            }
    table_request = {
            1: ['WAVELENGTH', 'FLUX', 'NET', 'BACKGROUND', 'DQ_WGT']
            }

    new_files_source = os.path.join(FILES_SOURCE, PID)
    files = find_files('*'+visit+'*x1d.fits*', data_dir=new_files_source)

    data_results = data_from_exposures(files,
                                           header_request=header_request,
                                           table_request=table_request)

    return data_results