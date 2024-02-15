"""
    IMPORTANT TO NOTE:
        This file was taken from the monitors file 'data_models.py' written by [author].

        The following code has been heavily edited and repurposed for the development 
        of the FUVTDS monitor by Jaq Hernandez.
"""

import pandas as pd
import os

from filesystem import find_files, data_from_exposures
from monitorframe.datamodel import BaseDataModel

FILES_SOURCE = '/grp/hst/cos2/cosmo'
PROGRAMS = '/Users/jhernandez/Desktop/fuvtds/monitor/fuvtds_analysis_list.dat'

def get_program_ids(pid_file):
    """Retrieve the program IDs from the given text file."""
    programs_df = pd.read_csv(pid_file, delim_whitespace=True)
    all_programs = []
    
    for col, col_data in programs_df.items():
        all_programs += col_data.to_numpy(dtype=str).tolist() 

    return (all_programs)

class FUVTDSModel(BaseDataModel):
    """DataModel for x1d files"""
    cosmo_layout = True
    files_source = FILES_SOURCE

    def get_new_data(self):
        """Set the model for what data is to be retrieved from each x1d file."""
        header_request = {
            0: ['ROOTNAME', 'SEGMENT', 'CENWAVE', 'TARGNAME', 'OPT_ELEM', 'LIFE_ADJ', 'EXPTYPE'],
            1: ['DATE-OBS']
        }
        table_request = {
            1: ['WAVELENGTH', 'FLUX', 'NET', 'BACKGROUND', 'DQ_WGT']
        }

        files = []
        program_ids = get_program_ids(PROGRAMS)

        for pid in program_ids:
            new_files_source = os.path.join(FILES_SOURCE, pid)
            files += find_files('*_x1d.fits*', data_dir=new_files_source)
        if self.model is not None:
            currently_ingested = [item.FILENAME for item in self.model.select(self.model.FILENAME)]

            for file in currently_ingested:
                files.remove(file)
        if not files: #No new files:
            return (pd.DataFrame())
        
        data_results = data_from_exposures(
            files,
            header_request=header_request,
            table_request=table_request
        )
        
        return (data_results)