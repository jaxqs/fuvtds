from astropy.io import fits
import numpy as np 
import pandas as pd
import os
import glob
from itertools import repeat
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map

def get_x1ds(cosmo_path, pids, pattern):
    # this will obtain the path to the x1d file, based off the data dir (cosmo) and pid
    total_path = os.path.join(cosmo_path, pids, pattern)
    path_list = glob.glob(total_path)
    return (path_list)

def which_files_to_run(cosmo_path, pids, pattern, csv_file):
    """
    Identifies the x1d files to run and the paths to them, based of the datadirectory (cosmo) and pid

    Args:
        cosmo_path: path to the directory where the data is stored
        pids: the program ids of the programs of interest
        pattern: the pattern of the file, in this case for x1d files
        csv_file: the csv file where we keep the info for the x1d files in question and where to find those files

    Returns:
        paths_to_run: the remaining paths to run that have not already been introduced into the csv file
    """
    with mp.Pool(16) as pool:
        x1d_paths = pool.starmap(get_x1ds, zip(repeat(cosmo_path), pids, repeat(pattern)))
    x1d_paths = [x for sublist in x1d_paths for x in sublist] # flatten list
    pool.terminate()

    if os.path.exists(csv_file):
        in_table = pd.read_csv(csv_file)
        paths_to_run = list(set(x1d_paths) - set(in_table.path))

    else:
        paths_to_run = list(x1d_paths)
    
    return (paths_to_run)

def get_x1ds_data(file_path):
    """
    Open up the x1d files to obtain the necessary data to populate the csv file.

    Args:
        file_path: the path to the file we want to open up
    
    Returns:
        x1d_table: DataFrame of the information we want from the x1d file
    """
    hdu = fits.open(file_path)
    exptime = hdu[1].header['exptime']
    
    # only want exposures that ran well and aren't WAVE files
    if (exptime != 0) & (hdu[0].header['targname'] != 'WAVE'):
        x1d_table = pd.DataFrame(
            {'rootname': [hdu[0].header['rootname']],
            'opt_elem': [hdu[0].header['opt_elem']],
            'cenwave': [hdu[0].header['cenwave']],
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

def update_inventory(new_table, csv_file):
    """
    This checks if 1) the csv file exists and 2) updates the new table to the csv file.
    If the csv file does not exist, then this will create the csv file.

    Args:
        new_table: the x1d_file created with the new x1dfiles that are not in the csv file currently.
        csv_file: the csv_file name to update/create.
    """
    if os.path.exists(csv_file):
        in_table = pd.read_csv(csv_file)
        new_table = pd.concat([in_table, new_table], ignore_index=True)
    else: 
        new_table.to_csv(csv_file)
        print(f'{csv_file} was created.')

def get_pids(pid_file):
    """
    From a file that contains the PIDs of all the FUVTDS Monitoring, put them into a list of strings.

    Args:
        pid_file: the file that contains the list of PIDs to be run.
    
    Returns:
        all_programs: a list of strings of all the PIDs.
    """
    programs_df = pd.read_csv(pid_file, delim_whitespace=True)
    all_programs = []
    for col, col_data in programs_df.items():
        all_programs += col_data.to_numpy(dtype=str).tolist()
    return (all_programs)

def analyze_files(cosmo_path, pid_file, pattern, csv_file):
    """
    Analyzes the x1d files and creates the inventory csv files to do analysis on.

    Args:
        cosmo_path: the path to the data directory
        pid_file: the file that contains the list of all the FUVTDS pids
        pattern: the file pattern, in this case x1ds.
        csv_file: the csv file that will store the files and file information
    """

    # get the list of string of pids
    pids = get_pids(pid_file)

    # the paths to the x1d files from cosmo, that have yet to be analyszed
    x1d_paths = which_files_to_run(cosmo_path, pids, pattern, csv_file)

    # the tables of information from the x1dfiles, in multiprocessing
    tables = process_map(get_x1ds_data, x1d_paths, max_workers=16, chunksize=10)

    # combine the many tables together into 1! table!
    tables = pd.concat(tables, ignore_index=True)

    # update or create the csv file that will store all this information
    update_inventory(tables, csv_file)

    # read in the csv file to be used for... something! we can use this command to do smth or the other lmfao
    csv = pd.read_csv(csv_file)
    return (csv)

if __name__ == "__main__":

    # change these as needed
    datadir = '/grp/hst/cos2/cosmo/'
    pattern = '*x1d.fits*'
    inventory = 'inventory.csv'
    pid_file = '/Users/jhernandez/Desktop/fuvtds/monitor/fuvtds_analysis_list.dat'

    csv = analyze_files(datadir, pid_file, pattern, inventory)