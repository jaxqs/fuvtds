"""
    IMPORTANT TO NOTE:
        This file was taken from the monitors file 'filesystem.py' written by [author].

        The following code has been lightly edited and repurposed for the development 
        of the FUVTDS monitor by Jaq Hernandez.
"""
import os
import numpy as np
import warnings
import abc
import dask

from glob import glob
from astropy.io import fits
from typing import Sequence, Union, List, Dict, Any

FILES_SOURCE = '/grp/hst/cos2/cosmo'
REQUEST = Dict[int, Sequence[str]]

class FileDataInterface(abc.ABC, dict):
    """Partial implementation for classes used to get data from COS FITS files that subclasses the python dictionary."""
    def __init__(self):
        super().__init__(self)

    @abc.abstractmethod
    def get_header_data(self, hdu: fits.HDUList, header_request: REQUEST, defaults: Dict[str, Any]):
        """Get header data."""
        pass

    @abc.abstractmethod
    def get_table_data(self, hdu: fits.HDUList, table_request: REQUEST):
        """Get table data."""
        pass


class FileData(FileDataInterface):
    """Class that acts as a dictionary, but with a constructor that grabs FITS file info from typical COS data
    products.
    """
    def __init__(self, hdu: fits.HDUList, header_request: REQUEST = None, table_request: REQUEST = None,
                 header_defaults: Dict[str, Any] = None, bytes_to_str: bool = True):
        """Initialize and create the possible corresponding spt file name."""
        super().__init__()

        if header_request:
            self.get_header_data(hdu, header_request, header_defaults)

        if table_request:
            self.get_table_data(hdu, table_request)

        if bytes_to_str:
            self._convert_bytes_to_strings()

    def _convert_bytes_to_strings(self):
        """Convert byte-string arrays to strings."""
        for key, value in self.items():
            if isinstance(value, np.ndarray):
                if value.dtype.char == 'S':
                    self[key] = value.astype(str)

    @classmethod
    def from_file(cls, filename, *args, **kwargs):
        with fits.open(filename) as hdu:
            return cls(hdu, *args, **kwargs)

    def get_header_data(self, hdu: fits.HDUList, header_request: REQUEST, header_defaults: dict = None):
        """Get header data."""
        for ext, keys in header_request.items():
            for key in keys:
                if header_defaults is not None and key in header_defaults:
                    self[key] = hdu[ext].header.get(key, default=header_defaults[key])

                else:
                    self[key] = hdu[ext].header[key]

    def get_table_data(self, hdu: fits.HDUList, table_request: REQUEST):
        """Get table data from the TableHDU."""
        for ext, keys in table_request.items():
            for key in keys:
                if key in self:
                    self[f'{key}_{ext}'] = hdu[ext].data[key]

                else:
                    self[key] = hdu[ext].data[key]

    def combine(self, other, right_name):
        """Combine two FileData dictionaries into one."""
        for key, value in other.items():
            if key in self:
                self[f'{right_name}_{key}'] = value

            else:
                self[key] = value

def find_files(file_pattern: str, data_dir: str = FILES_SOURCE, subdir_pattern: Union[str, None] = None) -> list:
    """Find COS data files from a source directory. The default is the cosmo data directory subdirectories layout
    pattern. A different subdirectory pattern can be used or
    """
    if subdir_pattern:
        return glob(os.path.join(data_dir, subdir_pattern, file_pattern))

    return glob(os.path.join(data_dir, file_pattern))

def get_exposure_data(filename: str, header_request: REQUEST = None, table_request: REQUEST = None,
                      header_defaults: Dict[str, Any] = None):
    """Get data requested from COS data and corresponding reference files."""

    try:
        with fits.open(filename) as hdu:
            if header_request or table_request:
                data = FileData(hdu, header_request, table_request, header_defaults)
                data['FILENAME'] = filename

    except OSError as e:
        warnings.warn(f'Bad file found: {filename}\n{str(e)}', Warning)

        return
    return data

def data_from_exposures(fitsfiles: List[str], header_request: REQUEST = None, table_request: REQUEST = None,
                        header_defaults: Dict[str, Any] = None):
    """Get requested data from COS files and their corresponding reference files in parallel."""
    delayed_results = [
        dask.delayed(get_exposure_data)(
            file,
            header_request,
            table_request,
            header_defaults
        ) for file in fitsfiles
    ]

    return [item for item in dask.compute(*delayed_results, scheduler='multiprocessing') if item is not None]