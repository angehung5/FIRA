"""
This script is used for gridded FRP pre-processing.
"""

import glob
import os
import warnings

import numpy as np
import pandas as pd
from netCDF4 import Dataset

warnings.simplefilter(action="ignore")


def rave_preprocessor(filename, time, lat_lim, lon_lim):
    namelist = pd.read_csv("./input/namelist", header=None, delimiter="=")
    namelist = namelist.rename(columns={0: "key", 1: "value"}).set_index("key")
    namelist = namelist.to_dict()["value"]
    path_frp = namelist["path_frp"]

    f_output = "./input/" + time + "/" + filename + "." + time + ".nc"

    year = time[:4]
    date = time[:8]
    hour = time[8:10]

    # ---- Reading Data ----
    # fname_beta = "Hourly_Emissions_3km_" + date + "0000_" + date + "2300.nc"
    # fname_v1 = "Hourly_Emissions_3km_"+ date + hour + "00_" + date + hour +"00.nc"
    if int(year) < 2024:
        fname = f"RAVE-HrlyEmiss-3km_v*r*_blend_s{date}0000000_e{date}2359590*.nc"
    else:
        fname = (
            f"RAVE-HrlyEmiss-3km_v*r*_blend_s{date}{hour}00000_e{date}{hour}59590*.nc"
        )
    f_ori = glob.glob(f"{path_frp}/{fname}")

    if len(f_ori) != 0:
        f_ori = f_ori[0]
        readin = Dataset(f_ori)
        yt = np.flip(readin["grid_latt"][:, 0])
        xt = readin["grid_lont"][0, :]
        xt[xt < 0] = xt[xt < 0] + 360

        if int(year) < 2024:
            tidx = int(hour)
        else:
            tidx = 0

        data = np.squeeze(readin["FRP_MEAN"][tidx, :, :])
        data = np.flipud(data)
        data = np.array(data)  # fill value = -1
        data[data == -1] = 0

        qa = readin["QA"][0, :, :]
        qa = np.flipud(qa)
        data[qa == 1] = 0  # use QA = 2 and 3 only

        index1 = np.squeeze(np.argwhere((yt >= lat_lim[0]) & (yt <= lat_lim[1])))
        index2 = np.squeeze(np.argwhere((xt >= lon_lim[0]) & (xt <= lon_lim[1])))

        if (index1[0] == 0) & (index2[0] == 0):
            yt = yt[index1[0] : index1[-1] + 2]
            xt = xt[index2[0] : index2[-1] + 2]
            data = data[index1[0] : index1[-1] + 2, index2[0] : index2[-1] + 2]
        elif index1[0] == 0:
            yt = yt[index1[0] : index1[-1] + 2]
            xt = xt[index2[0] - 1 : index2[-1] + 2]
            data = data[index1[0] : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]
        elif index2[0] == 0:
            yt = yt[index1[0] - 1 : index1[-1] + 2]
            xt = xt[index2[0] : index2[-1] + 2]
            data = data[index1[0] - 1 : index1[-1] + 2, index2[0] : index2[-1] + 2]
        else:
            yt = yt[index1[0] - 1 : index1[-1] + 2]
            xt = xt[index2[0] - 1 : index2[-1] + 2]
            data = data[index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]

        xt_grid, yt_grid = np.meshgrid(xt, yt)

        readin.close()
        del [readin, yt, xt, qa, index1, index2]

    else:
        return 1

    # ---- Write NetCDF File ----
    f = Dataset(f_output, "w")
    f.createDimension("time", 1)
    f.createDimension("nlat", data.shape[0])
    f.createDimension("nlon", data.shape[1])

    var_input = f.createVariable("frp", "float", ("time", "nlat", "nlon"))
    var_lat = f.createVariable("grid_lat", "float", ("nlat", "nlon"))
    var_lon = f.createVariable("grid_lon", "float", ("nlat", "nlon"))
    var_time = f.createVariable("time", str, ("time",))

    var_input[:] = data
    var_lat[:] = yt_grid
    var_lon[:] = xt_grid
    var_time[:] = np.array(time).astype(str)
    f.close()

    return 0


def abi_preprocessor(filename, time, lat_lim, lon_lim):
    namelist = pd.read_csv("./input/namelist", header=None, delimiter="=")
    namelist = namelist.rename(columns={0: "key", 1: "value"}).set_index("key")
    namelist = namelist.to_dict()["value"]
    path_frp = namelist["path_frp"]

    f_output = "./input/" + time + "/" + filename + "." + time + ".nc"

    date = time[:8]

    # ---- Reading Data ----
    f_ori = "ravelike_5min_emission_" + date + ".nc"
    f_ori = path_frp + "/" + f_ori

    if os.path.isfile(f_ori) is True:
        readin = Dataset(f_ori)
        yt = np.flip(readin["grid_latt"][:, 0])
        xt = readin["grid_lont"][0, :]
        xt[xt < 0] = xt[xt < 0] + 360

        time_series = pd.to_datetime(
            date + " 00:00", format="%Y%m%d %H:%M"
        ) + pd.to_timedelta(readin["time"][:], unit="min")
        time_idx = time_series.strftime("%Y%m%d%H%M") == time

        data = np.squeeze(readin["FRP_MEAN"][time_idx, :, :])
        data = np.flipud(data)
        data = np.array(data)
        data[data < 0] = 0

        index1 = np.squeeze(np.argwhere((yt >= lat_lim[0]) & (yt <= lat_lim[1])))
        index2 = np.squeeze(np.argwhere((xt >= lon_lim[0]) & (xt <= lon_lim[1])))

        if (index1[0] == 0) & (index2[0] == 0):
            yt = yt[index1[0] : index1[-1] + 2]
            xt = xt[index2[0] : index2[-1] + 2]
            data = data[index1[0] : index1[-1] + 2, index2[0] : index2[-1] + 2]
        elif index1[0] == 0:
            yt = yt[index1[0] : index1[-1] + 2]
            xt = xt[index2[0] - 1 : index2[-1] + 2]
            data = data[index1[0] : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]
        elif index2[0] == 0:
            yt = yt[index1[0] - 1 : index1[-1] + 2]
            xt = xt[index2[0] : index2[-1] + 2]
            data = data[index1[0] - 1 : index1[-1] + 2, index2[0] : index2[-1] + 2]
        else:
            yt = yt[index1[0] - 1 : index1[-1] + 2]
            xt = xt[index2[0] - 1 : index2[-1] + 2]
            data = data[index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]

        xt_grid, yt_grid = np.meshgrid(xt, yt)

        readin.close()
        del [readin, yt, xt, index1, index2]

    else:
        return 1

    # ---- Write NetCDF File ----
    f = Dataset(f_output, "w")
    f.createDimension("time", 1)
    f.createDimension("nlat", data.shape[0])
    f.createDimension("nlon", data.shape[1])

    var_input = f.createVariable("frp", "float", ("time", "nlat", "nlon"))
    var_lat = f.createVariable("grid_lat", "float", ("nlat", "nlon"))
    var_lon = f.createVariable("grid_lon", "float", ("nlat", "nlon"))
    var_time = f.createVariable("time", str, ("time",))

    var_input[:] = data
    var_lat[:] = yt_grid
    var_lon[:] = xt_grid
    var_time[:] = np.array(time).astype(str)
    f.close()

    return 0
