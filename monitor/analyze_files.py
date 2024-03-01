from astropy.io import fits
import pandas as pd
import os
import glob
from itertools import repeat
import multiprocessing as mp
from tqdm.contrib.concurrent import process_map

def bad_list(grating): 
    # check against inventory dot txt to remove th G140L LP4 rootnames.
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
                          'ldqj59k3q', 'ldv007paq', 'ldv010evq', 'lefe03grq', 'lbxmt1nzq', 'lbxmt3lwq'],
                'G160M': ['lbb917kfq', 'ldv006lqq', 'ldv007pcq', 'ler106dqq', 'lf2002e3q', 'lf201dsfq', 
                          'lf2006lsq', 'lf205bahq', 'lf205dmdq', 'lf2056ftq', 'lf207bkkq', 'lf2012drq', 
                          'lf2011idq', 'lf4g02jnq', 'lf4g1bs4q', 'ldqj05y0q', 'ldqj08j3q', 'ldqj13e9q', 
                          'ldqj12e8q', 'ldqj56a9q', 'ldqj57txq', 'ldqj58hgq', 'ldv006lsq', 'ldqj59jzq', 
                          'ldv007peq', 'ldv008ojq', 'ldv010flq', 'ler106dsq', 'lf2002e6q', 'lf201dshq', 
                          'lf2006liq', 'lf2006luq', 'lf205bakq', 'lf205bauq', 'lf205dmfq', 'lf2056g5q', 
                          'lf207bleq', 'lf2012dtq', 'lf2011ifq', 'lf4g02jpq', 'lf4g1bs6q', 'ldqj05y2q', 
                          'ldqj08j5q', 'ldqj13ebq', 'ldqj12eaq', 'ldqj56abq', 'ldqj57tzq', 'ldqj58hjq', 
                          'ldv006lvq', 'ldqj59k1q', 'ldv007pgq', 'ldv008olq', 'ldv010fnq', 'ler106doq', 
                          'ler106dwq', 'lf203bn8q', 'lf2004j0q', 'lf2006lqq', 'lf205bb8q', 'lf207bkoq', 
                          'lf208bktq', 'lf2009i4q', 'lf2111zuq', 'lf218blhq', 'lf203bn6q', 'ler106duq', 
                          'lf4h1bbwq', 'lf218bljq', 'ler106dkq', 'lf4h04wyq', 'lbxm02agq', 'lf2004iyq', 
                          'lf208bkrq', 'lbxm02bxq', 'lf4h04x0q', 'lf2009i2q', 'lf2111zoq', 'lf2006lmq',
                          'lbxm02ayq', 'lf205bb1q', 'lf2109zlq', 'lf207bkmq', 'lf4h1bbyq']} 

    return(bad_roots[grating])

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
        paths_to_run = list(set(x1d_paths) - set(in_table))

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

    #bad_targs = ['WAVE', 'LDS749B']
    #bad_items = bad_list(hdu[0].header['opt_elem'])
    
    def c1280_check(cenwave, fppos, expstart):
        good = False
        if (fppos == 3) & (cenwave != 1280):
            good = True
        elif (cenwave == 1280) & (fppos == 4) & (expstart < 56130.0):
            good = True
        elif (cenwave == 1280) & (fppos == 3) & (expstart > 56130.0):
            good = True
        return (good)

    criteria = {
        'exptime': exptime !=0,
        'bad_targs': hdu[0].header['targname'] not in ['WAVE', 'LDS749B'],
        'bad_items': hdu[0].header['rootname'] not in bad_list(hdu[0].header['opt_elem']),
        'fppos_check': c1280_check(
            hdu[0].header['cenwave'],
            hdu[0].header['fppos'],
            hdu[1].header['expstart']) == True,
        'wl': len(hdu[1].data['wavelength']) != 0,
        'lp4': (hdu[0].header['opt_elem'] != 'G160M') | ((hdu[0].header['life_adj'] != 4) | (hdu[1].header['date-obs'] < '2022-10-01'))
    }
    if (criteria['exptime']) & (criteria['bad_targs']) & (criteria['bad_items']) & (criteria['fppos_check']) & (criteria['wl']) & (criteria['lp4']):
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
            'file_path': [file_path],
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
    new_table = new_table.sort_values(by=['date-obs'], ignore_index=True)
    if os.path.exists(csv_file):
        new_table.to_csv(csv_file, mode='w+')
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
    if len(tables) !=0:
        update_inventory(tables, csv_file)

if __name__ == "__main__":

    # change these as needed
    datadir = '/grp/hst/cos2/cosmo/'
    pattern = '*x1d.fits*'
    inventory = 'inventory.csv'
    pid_file = '/Users/jhernandez/Desktop/fuvtds/monitor/fuvtds_analysis_list.dat'

    analyze_files(datadir, pid_file, pattern, inventory)