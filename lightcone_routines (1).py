import numpy as N
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import os
from collections import OrderedDict


#lightcone_data_path = "/data/dega1/iprn/galform_out/r635/mini/Gonzalez17.PAUS.MilliGas62/ivol_0/galaxies.hdf5"
#lightcone_figure_path = '/cosma/home/iprn/python/DESIlightcone/figs'

def read_redshift_data(datapath=None):
    filename = datapath
    if (os.path.isfile(filename) == False):
        print "file not found:", filename
        return pd.DataFrame()     
    f = h5py.File(datapath, 'r')
    redshifts = f["Redshifts"]#[()]
    return pd.DataFrame({"redshifts":redshifts})

def select_snapshots(redshift = float):
    redshift_data_path = "/data/dega1/iprn/galform_out/r635/mini/Gonzalez17.PAUS.MilliGas62/ivol_0/galaxies.hdf5"
    redshift_data_tmp = read_redshift_data(datapath = redshift_data_path)
    redshift_data_dict = OrderedDict()
    snapshot_number = 1

    #Select desired redshift
    #redshift = 0.7

    #Creates dictionary containing outputs and corresponding redshift values
    for ii in range(23):
        snapshot = "Output%03d" % (snapshot_number)
        redshift_data_dict[snapshot] = float(redshift_data_tmp.iloc[ii]['redshifts'])
        snapshot_number = snapshot_number + 1

    #Runs through dictionary to find snapshots given redshift is between and calculates
    #corresponding beginning and end snapshots for other merger trees
    for key, value in reversed(redshift_data_dict.items()):
        if value <= redshift:
            snapshot1_62_id = int(key[-3:])
            snapshot1_123_id = (snapshot1_62_id * 2) - 1
            snapshot1_245_id = (snapshot1_123_id * 2) - 1
            snapshot1_489_id = (snapshot1_245_id * 2) - 1           
            snapshot1_977_id = (snapshot1_489_id * 2) - 1
            break
    for key, value in redshift_data_dict.items():
        if value >= redshift:
            snapshot2_62_id = int(key[-3:])
            snapshot3_123_id = (snapshot2_62_id * 2) - 1
            snapshot5_245_id = (snapshot3_123_id * 2) - 1
            snapshot9_489_id = (snapshot5_245_id * 2) - 1                     
            snapshot17_977_id = (snapshot9_489_id * 2) - 1
            break            

    #Finds the intermediate snapshots adding them to a dictionary of lists
    required_snapshot_ids = OrderedDict()
    all_62_ids = range(snapshot1_62_id, (snapshot2_62_id+1))
    required_snapshot_ids[62] = all_62_ids
    all_123_ids = range(snapshot1_123_id, (snapshot3_123_id+1))
    required_snapshot_ids[123] = all_123_ids
    all_245_ids = range(snapshot1_245_id, (snapshot5_245_id+1))
    required_snapshot_ids[245] = all_245_ids
    all_489_ids = range(snapshot1_489_id, (snapshot9_489_id+1))
    required_snapshot_ids[489] = all_489_ids
    all_977_ids = range(snapshot1_977_id, (snapshot17_977_id+1))
    required_snapshot_ids[977] = all_977_ids

    #Formats the integers into strings compatible for use within datapath
    for key in required_snapshot_ids:
        required_snapshot_ids[key] = ["Output%03d" % (value) for value in required_snapshot_ids[key]]
    return required_snapshot_ids

    
def read_snapshot_data(snapshot=None, datapath=None):
    filename = datapath
    if (os.path.isfile(filename) == False):
        print "file not found:", filename
        return pd.DataFrame()     
    f = h5py.File(datapath, 'r')
    GalaxyID = f["%s/GalaxyID" % (snapshot)][()]
    SubhaloIndex = f["%s/SubhaloIndex" % (snapshot)][()]
    ParticleID = f["%s/ParticleID" % (snapshot)][()]
    DescendantID = f["%s/DescendantID" % (snapshot)][()]
    FirstProgenitorID = f["%s/FirstProgenitorID" % (snapshot)][()]
    SubhaloID = f["%s/SubhaloID" % (snapshot)][()]
    DHaloID = f["%s/DHaloID" % (snapshot)][()]
    mhalo = f["%s/mhalo" % (snapshot)][()]
    mhhalo = f["%s/mhhalo" % (snapshot)][()]
    redshift = f["%s/redshift" % (snapshot)][()]
    galtype = f["%s/type" % (snapshot)][()]
    xgal = f["%s/xgal" % (snapshot)][()]
    ygal = f["%s/ygal" % (snapshot)][()]
    zgal = f["%s/zgal" % (snapshot)][()]
    vxgal = f["%s/vxgal" % (snapshot)][()]
    vygal = f["%s/vygal" % (snapshot)][()]
    vzgal = f["%s/vzgal" % (snapshot)][()]
 
    return pd.DataFrame({"GalaxyID":GalaxyID, "SubhaloIndex":SubhaloIndex, "SubhaloID":SubhaloID, "ParticleID":ParticleID,
                         "DescendantID":DescendantID, "FirstProgenitorID":FirstProgenitorID, "DHaloID":DHaloID, 
                         "mhalo":mhalo,"mhhalo":mhhalo,"redshift":redshift, "galtype":galtype, 
                         "xgal":xgal,"ygal":ygal,"zgal":zgal,"vxgal":vxgal, "vygal":vygal,"vzgal":vzgal})

