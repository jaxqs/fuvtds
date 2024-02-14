"""
THIS CODE IS NOT PART OF THE FUV TDS MONITOR CODING.
THIS WAS WRITTEN TO OBTAIN THE MANUALLY EXCLUDED DATASETS FOUND IN INVENTORY.TXT IN ORDER TO CREATE
A BAD ITEM FUNCTION IN THE ANALYSIZE_FILES.PY 

AGAIN:
    THIS IS NOT PART OF THE FUV TDS MONITOR CODE AND THUS CAN BE EXCLUDED
"""

from astropy.time import Time
import numpy as np
from astropy.io import ascii
import pandas as pd

inventory_1055 = ascii.read('inventory.txt', delimiter="\s", header_start=2, data_start=3, data_end=68)
inventory_1096 = ascii.read('inventory.txt', delimiter="\s", header_start=70, data_start=71, data_end=184)
inventory_1222 = ascii.read('inventory.txt', delimiter="\s", header_start=186, data_start=187, data_end=268)
inventory_1291 = ascii.read('inventory.txt', delimiter="\s", header_start=270, data_start=271, data_end=413)
inventory_1309 = ascii.read('inventory.txt', delimiter="\s", header_start=415, data_start=416, data_end=456)
inventory_1327 = ascii.read('inventory.txt', delimiter="\s", header_start=458, data_start=459, data_end=577)
inventory_800  = ascii.read('inventory.txt', delimiter="\s", header_start=579, data_start=580, data_end=616)
inventory_1105 = ascii.read('inventory.txt', delimiter="\s", header_start=618, data_start=619, data_end=741)
inventory_1280 = ascii.read('inventory.txt', delimiter="\s", header_start=743, data_start=744, data_end=885)
inventory_1533 = ascii.read('inventory.txt', delimiter="\s", header_start=887, data_start=888, data_end=963)
inventory_1577 = ascii.read('inventory.txt', delimiter="\s", header_start=965, data_start=966, data_end=1150)
inventory_1600 = ascii.read('inventory.txt', delimiter="\s", header_start=1152, data_start=1153, data_end=1188)
inventory_1611 = ascii.read('inventory.txt', delimiter="\s", header_start=1190, data_start=1191, data_end=1245)
inventory_1623 = ascii.read('inventory.txt', delimiter="\s", header_start=1247, data_start=1248, data_end=1456)

LP3_LP4_switch = Time('2017-10-02').decimalyear#-0.1
rootname_1055 = inventory_1055['Rootname'][Time(inventory_1055['date-obs']).decimalyear>LP3_LP4_switch]
rootname_1096 = inventory_1096['Rootname'][((inventory_1096['targname']=='GD71'))] # Calibrate all
rootname_1096_wave = inventory_1096['Rootname'][((inventory_1096['targname']=='WAVE'))] # Calibrate all
rootname_1222 = inventory_1222['Rootname'][Time(inventory_1222['date-obs']).decimalyear>LP3_LP4_switch]
rootname_1291 = inventory_1291['Rootname'][Time(inventory_1291['date-obs']).decimalyear>LP3_LP4_switch]
rootname_1327 = inventory_1327['Rootname'][Time(inventory_1327['date-obs']).decimalyear>LP3_LP4_switch]
rootname_800 = inventory_800['Rootname'][Time(inventory_800['date-obs']).decimalyear>LP3_LP4_switch]
rootname_1105 = inventory_1105['Rootname'][Time(inventory_1105['date-obs']).decimalyear>LP3_LP4_switch]
rootname_1280 = inventory_1280['Rootname'][Time(inventory_1280['date-obs']).decimalyear>LP3_LP4_switch]
rootname_1533A = inventory_1533['Rootname'][((inventory_1533['targname']=='GD71') & (Time(inventory_1533['date-obs']).decimalyear>LP3_LP4_switch))]
rootname_1533B = inventory_1533['Rootname'][((inventory_1533['targname']=='WD0308-565') & (Time(inventory_1533['date-obs']).decimalyear>LP3_LP4_switch))]
rootname_1577A = inventory_1577['Rootname'][((inventory_1577['targname']=='GD71') & (Time(inventory_1577['date-obs']).decimalyear>LP3_LP4_switch))]
rootname_1577A_wave = inventory_1577['Rootname'][((inventory_1577['targname']=='WAVE') & (Time(inventory_1577['date-obs']).decimalyear>LP3_LP4_switch))]
rootname_1577B = inventory_1577['Rootname'][((inventory_1577['targname']=='WD0308-565') & (Time(inventory_1577['date-obs']).decimalyear>LP3_LP4_switch))]
rootname_1623A = inventory_1623['Rootname'][((inventory_1623['targname']=='GD71') & (Time(inventory_1623['date-obs']).decimalyear>LP3_LP4_switch))]
rootname_1623B = inventory_1623['Rootname'][((inventory_1623['targname']=='WD0308-565') & (Time(inventory_1623['date-obs']).decimalyear>LP3_LP4_switch))]
#Temporarily remove rootname_1533B first obs ldsi51kaq (1533 fluxes and flats program from testing, won’t calibrate due to asn=None). Same for ldsi02u2q.
rootname_1533A = rootname_1533A[1:]
rootname_1533B = rootname_1533B[1:]
rootname_800 = rootname_800[1:]

inventory = pd.read_csv('inventory.csv')

rootnames = np.array(inventory.loc[(inventory['cenwave'] == 1623) & (inventory['date-obs'] > '2017-10-02'), ['rootname']]).flatten()

for row in rootnames:
    if (row not in rootname_1623A) & ((row not in rootname_1623B)):
        print(f"{row} {np.array(inventory['file_path'][inventory['rootname'] == row])}")
    #else:
    #    print('yes!')
