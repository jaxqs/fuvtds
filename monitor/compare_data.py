import glob
import numpy as np
import pandas as pd
from astropy.io import fits

original = np.array(glob.glob('/grp/hst/cos2/fuv_tds_data_Mar2022/*.fits'))
mine = np.array(glob.glob('/grp/hst/cos2/fuv_tds_2024/fuvtds_analysis/data/*.fits'))

original_names = []
for name in original:
    fn = name.split('/')
    original_names.append(fn[-1])

mine_names = []
for name in mine:
    fn = name.split('/')
    mine_names.append(fn[-1])

# This checks if I contain a dataset that MAR2022 does not contain
counter = 0
for i, row in enumerate(mine_names):
    if row not in original_names:
        counter = counter + 1
        print(row, i)
print(counter)

# This checks if MAR2022 contain a dataset that I do not contain
"""
counter = 0 
for i, row in enumerate(original_names):
    if row not in mine_names:
        counter = counter + 1
        print(row)
print(counter)


tables = []
for i, row in enumerate(original_names):
    if row not in mine_names:
        hdu = fits.open(original[i])
        x1d_table = pd.DataFrame(
            {
                'rootname': [hdu[0].header['rootname']],
                'PID': [hdu[0].header['proposid']],
                'exptime': [hdu[1].header['exptime']],
                'grating': [hdu[0].header['opt_elem']],
                'cenwave': [hdu[0].header['cenwave']],
                'segment': [hdu[0].header['segment']],
                'fppos': [hdu[0].header['fppos']],
                'lp': [hdu[0].header['life_adj']],
                'date-obs': [hdu[1].header['date-obs']],
                'targname': [hdu[0].header['targname']],
                'length_wl':[len(hdu[1].data['wavelength'])]
            }
        )
        tables.append(x1d_table)
        hdu.close()

tables = pd.concat(tables, ignore_index=True)
tables.to_csv('missing_x1d_in_mine.csv')
"""