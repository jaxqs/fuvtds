from astropy.io import fits
import numpy as np 
import pandas as pd
import os
import glob
from itertools import repeat
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map

def get_x1ds(cosmo_path, pids, pattern):
    total_path = os.path.join(cosmo_path, pids, pattern)
    path_list = glob.glob(total_path)
    return (path_list)

def which_files_to_run(cosmo_path, pids, pattern, csv_file):
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
    hdu = fits.open(file_path)
    exptime = hdu[1].header['exptime']
    
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
    if os.path.exists(csv_file):
        in_table = pd.read_csv(csv_file)
        new_table = pd.concat([in_table, new_table])
    else: 
        new_table.to_csv(csv_file)
        print(f'{csv_file} was created.')

def get_pids(pid_file):
    programs_df = pd.read_csv(pid_file, delim_whitespace=True)
    all_programs = []
    for col, col_data in programs_df.items():
        all_programs += col_data.to_numpy(dtype=str).tolist()
    return (all_programs)

def analyze_files(cosmo_path, pid_file, pattern, csv_file):

    pids = get_pids(pid_file)

    x1d_paths = which_files_to_run(cosmo_path, pids, pattern, csv_file)

    tables = process_map(get_x1ds_data, x1d_paths, max_workers=16, chunksize=10)

    tables = pd.concat(tables, ignore_index=True)
    update_inventory(tables, csv_file)

    csv = pd.read_csv(csv_file)
    #csv.index = pd.DatetimeIndex(csv['date-obs'])
    return (csv)

if __name__ == "__main__":

    datadir = '/grp/hst/cos2/cosmo/'
    pattern = '*x1d.fits*'
    inventory = 'inventory.csv'
    pid_file = '/Users/jhernandez/Desktop/fuvtds/monitor/fuvtds_analysis_list.dat'

    csv = analyze_files(datadir, pid_file, pattern, inventory)