"""
This script is used to generate model input files. Pre-process gridded fire map needed.
"""

import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from metpy.calc import wind_direction
from metpy.units import units
from netCDF4 import Dataset
from scipy import ndimage
from scipy.interpolate import griddata

from fuel_consumption_cal import cr_cal_by_frp
from mete_reader import gfs_reader, hrrr_reader, rap_reader, wofs_reader

warnings.simplefilter(action="ignore")


def file_finder(item, date, initial_hour, fcst_hour, fcst_freq=1):
    namelist = pd.read_csv("./input/namelist", header=None, delimiter="=")
    namelist = namelist.rename(columns={0: "key", 1: "value"}).set_index("key")
    namelist = namelist.to_dict()["value"]
    mete_source = int(namelist["mete_source"])  # 0: hrrr, 1: rap, 2: gfs, 3: wofs

    if item == "ast":
        return namelist["path_ast"] + "/canopy_surface_type.2020.0.01.nc"
    elif item == "ch":
        return namelist["path_ch"] + "/canopy_height.2020.0.01.nc"
    elif item == "pai":
        if int(date[:4]) <= 2020:
            return (
                namelist["path_pai"] + "/canopy_leaf_area_index." + "2020" + ".0.01.nc"
            )
        elif int(date[:4]) >= 2024:
            return (
                namelist["path_pai"] + "/canopy_leaf_area_index." + "2024" + ".0.01.nc"
            )
        else:
            return (
                namelist["path_pai"]
                + "/canopy_leaf_area_index."
                + date[:4]
                + ".0.01.nc"
            )
    elif (item == "t2m") or (item == "vpd") or (item == "wd") or (item == "ws"):
        if mete_source == 0:
            return (
                namelist["path_mete"]
                + "/hrrr.t"
                + initial_hour
                + "z.wrfsfcf"
                + ("%02d" % fcst_hour)
                + ".grib2"
            )
        elif mete_source == 1:
            dstr = datetime(int(date[:4]), int(date[4:6]), int(date[6:]))
            return (
                namelist["path_mete"]
                + "/"
                + dstr.strftime("%y%j")
                + ("%02d" % fcst_hour)
                + "000000"
            )
        elif mete_source == 2:
            return (
                namelist["path_mete"]
                + "/gfs.t"
                + initial_hour
                + "z.sfcf"
                + ("%03d" % fcst_hour)
                + ".nc"
            )
        elif mete_source == 3:
            dstr = datetime(int(date[:4]), int(date[4:6]), int(date[6:]), initial_hour)
            diff = timedelta(minutes=fcst_hour * fcst_freq)
            return (
                namelist["path_mete"]
                + "/wofs-e_ENV_"
                + ("%02d" % fcst_hour)
                + "_"
                + dstr.strftime("%Y%m%d_%H%M")
                + "_"
                + (dstr + diff).strftime("%H%M")
                + ".nc"
            )
    elif item == "prate":
        if mete_source == 0:
            return (
                namelist["path_mete"]
                + "/hrrr.t"
                + initial_hour
                + "z.wrfsfcf"
                + ("%02d" % (fcst_hour + 1))
                + ".grib2"
            )
        elif mete_source == 1:
            dstr = datetime(int(date[:4]), int(date[4:6]), int(date[6:]))
            if fcst_hour == 23:
                return (
                    namelist["path_mete"]
                    + "/"
                    + (dstr + pd.Timedelta(days=1)).strftime("%y%j")
                    + ("%02d" % 0)
                    + "000000"
                )
            else:
                return (
                    namelist["path_mete"]
                    + "/"
                    + dstr.strftime("%y%j")
                    + ("%02d" % (fcst_hour + 1))
                    + "000000"
                )
        elif mete_source == 2:
            return (
                namelist["path_mete"]
                + "/gfs.t"
                + initial_hour
                + "z.sfcf"
                + ("%03d" % fcst_hour)
                + ".nc"
            )
        elif mete_source == 3:
            dstr = datetime(int(date[:4]), int(date[4:6]), int(date[6:]), initial_hour)
            diff0 = timedelta(minutes=(fcst_hour - 1) * fcst_freq)
            diff1 = timedelta(minutes=fcst_hour * fcst_freq)
            file0 = namelist["path_mete"]
            +"/wofs-e_ENV_"
            +("%02d" % fcst_hour)
            +"_"
            +dstr.strftime("%Y%m%d_%H%M")
            +"_"
            +(dstr + diff0).strftime("%H%M")
            +".nc"
            file1 = namelist["path_mete"]
            +"/wofs-e_ENV_"
            +("%02d" % fcst_hour)
            +"_"
            +dstr.strftime("%Y%m%d_%H%M")
            +"_"
            +(dstr + diff1).strftime("%H%M")
            +".nc"
            return (file0, file1)
    elif item == "soilw":
        if mete_source == 0:
            return (
                namelist["path_mete"]
                + "/hrrr.t"
                + initial_hour
                + "z.wrfprsf"
                + ("%02d" % fcst_hour)
                + ".grib2"
            )
        elif mete_source == 1:
            dstr = datetime(int(date[:4]), int(date[4:6]), int(date[6:]))
            return (
                namelist["path_mete"]
                + "/"
                + dstr.strftime("%y%j")
                + ("%02d" % fcst_hour)
                + "000000"
            )
        elif mete_source == 2:
            return (
                namelist["path_mete"]
                + "/gfs.t"
                + initial_hour
                + "z.sfcf"
                + ("%03d" % fcst_hour)
                + ".nc"
            )
        elif mete_source == 3:
            dstr = datetime(int(date[:4]), int(date[4:6]), int(date[6:]), initial_hour)
            diff = timedelta(minutes=fcst_hour * fcst_freq)
            return (
                namelist["path_mete"]
                + "/wofs-e_ENV_"
                + ("%02d" % fcst_hour)
                + "_"
                + dstr.strftime("%Y%m%d_%H%M")
                + "_"
                + (dstr + diff).strftime("%H%M")
                + ".nc"
            )


def mapping(xgrid, ygrid, data, xdata, ydata, map_method, fill_value):
    output = griddata(
        (xdata, ydata), data, (xgrid, ygrid), method=map_method, fill_value=fill_value
    )
    return output


def wind_conversion(
    LAT, U, V
):  # from HRRR FAQ (https://rapidrefresh.noaa.gov/faq/HRRR.faq.html)
    rotcon_p = 0.622515
    lat_tan_p = 38.5

    angle = rotcon_p * (LAT - lat_tan_p) * 0.017453  # LAMBERT CONFORMAL PROJECTION
    sinx2 = np.sin(angle)
    cosx2 = np.cos(angle)

    U_new = cosx2 * U + sinx2 * V
    V_new = (-1) * sinx2 * U + cosx2 * V
    return U_new, V_new


def BAI_reader(yt_lim, xt_lim, yt_target, xt_target):
    dx = 1000  # km -> m

    readin = Dataset("./fix/budiling_area_index_2017_2022.nc")
    yt = readin["lat"][:]
    xt = readin["lon"][:]
    xt[xt < 0] = xt[xt < 0] + 360
    index1 = np.squeeze(np.argwhere((yt >= yt_lim[0]) & (yt <= yt_lim[1])))
    index2 = np.squeeze(np.argwhere((xt >= xt_lim[0]) & (xt <= xt_lim[1])))

    yt = yt[index1[0] - 1 : index1[-1] + 2]
    xt = xt[index2[0] - 1 : index2[-1] + 2]

    area = readin["area"][
        index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2
    ]
    height = readin["height_mean"][
        index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2
    ]
    qa = readin["qa"][index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]
    height[qa < 0.5] = 0  # remove low quality data
    total_area = np.zeros(area.shape)
    total_area[(area > 0) & (height != 0)] = area[(area > 0) & (height != 0)] + (
        4
        * np.sqrt(area[(area > 0) & (height != 0)])
        * height[(area > 0) & (height != 0)]
    )
    total_area[(area > 0) & (height == 0)] = 5 * area[(area > 0) & (height == 0)]
    data = total_area / (dx**2)  # area index
    del [area, height, qa, total_area]

    xt_grid, yt_grid = np.meshgrid(xt, yt)
    data_grid = mapping(
        yt_target,
        xt_target,
        data.ravel(),
        yt_grid.ravel(),
        xt_grid.ravel(),
        "nearest",
        np.nan,
    )
    data_grid[data_grid < 0] = 0

    return data_grid


def PAI_cal(LAI, vtype):
    # follow the approach used in canopy-app
    # PAI = LAI/(1-alpha), alpha = wood-to-total area ratio (Fang et al., 2019)
    # https://doi.org/10.1029/2018RG000608
    LAI[(vtype == 1) | (vtype == 2)] = LAI[(vtype == 1) | (vtype == 2)] / (1 - 0.1231)
    LAI[(vtype == 3) | (vtype == 4)] = LAI[(vtype == 3) | (vtype == 4)] / (1 - 0.185)
    LAI[vtype == 5] = LAI[vtype == 5] / (1 - 0.14)
    LAI[(vtype >= 6) & (vtype <= 10)] = LAI[(vtype >= 6) & (vtype <= 10)] / (1 - 0.26)
    LAI[(vtype == 12) | (vtype == 14)] = LAI[(vtype == 12) | (vtype == 14)] / (1 - 0.26)
    LAI[(vtype == 18) | (vtype == 19)] = LAI[(vtype == 18) | (vtype == 19)] / (1 - 0.26)
    return LAI


def spatial_weight(fsize):
    dx, dy = np.meshgrid(
        np.arange(-1 * fsize, fsize + 0.1, 1), np.arange(-1 * fsize, fsize + 0.1, 1)
    )
    dis = np.sqrt(dx**2 + dy**2)

    dx = units.Quantity(dx, "m/s")
    dy = units.Quantity(dy, "m/s")
    degree = wind_direction(dx, dy, convention="to")
    degree = np.array(degree)

    dis = np.expand_dims(dis, axis=-1)
    degree = np.expand_dims(degree, axis=-1)
    return dis, degree


def normalization(var, NN):
    a = []
    b = []
    for i in np.arange(NN):
        a = np.append(a, np.nanmax(var[:, :, :, i]) - np.nanmin(var[:, :, :, i]))
        b = np.append(b, np.nanmin(var[:, :, :, i]))
    return a, b


def add_manual_fire(frp_map, lat_map, lon_map, flat, flon, flvl):
    for i in np.arange(len(flvl)):
        dis = (lat_map - flat[i]) ** 2 + (lon_map - flon[i]) ** 2

        if frp_map[dis == np.min(dis)] == 0:
            frp_map[dis == np.min(dis)] = flvl[i]

    return frp_map


def main_driver(initial_hour, forecast_hour, f_input, f_output, lat_lim, lon_lim):
    namelist = pd.read_csv("./input/namelist", header=None, delimiter="=")
    namelist = namelist.rename(columns={0: "key", 1: "value"}).set_index("key")
    namelist = namelist.to_dict()["value"]

    mete_source = int(namelist["mete_source"])  # 0: hrrr, 1: rap, 2: gfs, 3: wofs
    time_freq = int(namelist["time_freq"])
    init_fuel = float(namelist["fuel_max"])
    manual_fire = namelist["man_fire"] in ["True", "true"]
    fuel_consume = namelist["fuel_consume"] in ["True", "true"]

    # ---- Global Settings ----
    # constants
    fsize = 5  # extend fire grid

    # variable list
    firelist = ["frp"]
    geolist = ["ast"]
    veglist = ["ch", "pai", "fuel"]
    metlist = ["t2m", "vpd", "wd", "ws", "prate", "soilw"]
    INPUTLIST = firelist + geolist + veglist + metlist

    # ---- Reading Fire Map & Input Settings ----
    if forecast_hour == 0:  # initial fire
        readin = Dataset(f_input)
        time = readin["time"][0]
        LAT = readin["grid_lat"][:]
        LON = readin["grid_lon"][:]
        FRP = readin["frp"][0, :, :]
        readin.close()
        del readin

        if manual_fire:
            man_lat = np.array([float(s) for s in namelist["man_lat"].split(",")])
            man_lon = np.array([float(s) for s in namelist["man_lon"].split(",")])
            man_lvl = np.array([float(s) for s in namelist["man_lvl"].split(",")])
            add_manual_fire(FRP, LAT, LON, man_lat, man_lon, man_lvl)
            print("---- Warning!!! Manual fire added!!!")
            del [man_lat, man_lon, man_lvl]
    else:
        readin = Dataset(f_input)
        time = readin["time"][0]
        LAT = readin["grid_lat"][:]
        LON = readin["grid_lon"][:]
        FRP = readin["grid_predic"][0, :, :]
        readin.close()
        del readin

    NN = len(INPUTLIST)
    tt = time  # yyyymmddHHMM
    dd = time[:8]  # yyyymmdd
    mm = time[4:6]  # mm
    hh = time[8:10]  # HH

    INPUT = np.empty([LAT.shape[0], LAT.shape[1], NN])
    print("---- Selecting data...", dd, hh + "Z")

    # ---- Reading Input Variables ----
    # frp, lat, lon
    INPUT[:, :, INPUTLIST.index("frp")] = np.copy(FRP)

    # ast
    filename = file_finder("ast", dd, initial_hour, forecast_hour)

    if os.path.isfile(filename) is False:
        print("---- Incorrect input file:", "ast", "Terminated!")
        print(filename)
        return 1

    readin = Dataset(filename)
    yt = readin["lat"][:]
    xt = readin["lon"][:]
    yt = np.round(yt, 3)
    xt = np.round(xt, 3)
    xt[xt < 0] = xt[xt < 0] + 360
    index1 = np.squeeze(np.argwhere((yt >= lat_lim[0]) & (yt <= lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt >= lon_lim[0]) & (xt <= lon_lim[1])))

    data = np.squeeze(readin["canopy_vegetation_type"][:])

    # ast, hour x lat x lon
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
    data_grid = mapping(
        LAT,
        LON,
        data.ravel(),
        yt_grid.ravel(),
        xt_grid.ravel(),
        "nearest",
        np.nan,
    )

    INPUT[:, :, INPUTLIST.index("ast")] = np.copy(data_grid)

    readin.close()
    del [filename, readin, yt, xt, yt_grid, xt_grid, index1, index2, data, data_grid]

    # ch
    filename = file_finder("ch", dd, initial_hour, forecast_hour)

    if os.path.isfile(filename) is False:
        print("---- Incorrect input file:", "ch", "Terminated!")
        print(filename)
        return 1

    readin = Dataset(filename)
    yt = readin["lat"][:]
    xt = readin["lon"][:]
    yt = np.round(yt, 3)
    xt = np.round(xt, 3)
    xt[xt < 0] = xt[xt < 0] + 360
    index1 = np.squeeze(np.argwhere((yt >= lat_lim[0]) & (yt <= lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt >= lon_lim[0]) & (xt <= lon_lim[1])))

    data = np.squeeze(readin["canopy_height"][:])

    # ch, lat x lon
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
    data_grid = mapping(
        LAT, LON, data.ravel(), yt_grid.ravel(), xt_grid.ravel(), "linear", np.nan
    )
    data_grid[data_grid < 0] = np.nan
    data_grid[np.isnan(data_grid)] = 0

    INPUT[:, :, INPUTLIST.index("ch")] = np.copy(data_grid)

    readin.close()
    del [filename, readin, yt, xt, yt_grid, xt_grid, index1, index2, data, data_grid]

    # pai
    filename = file_finder("pai", dd, initial_hour, forecast_hour)

    if not os.path.isfile(filename):
        print("---- Incorrect input file:", "pai", "Terminated!")
        print(filename)
        exit()

    readin = Dataset(filename)
    yt = readin["lat"][:]
    xt = readin["lon"][:]
    index1 = np.squeeze(np.argwhere((yt >= lat_lim[0]) & (yt <= lat_lim[1])))
    index2 = np.squeeze(np.argwhere((xt >= lon_lim[0]) & (xt <= lon_lim[1])))

    tidx = int(mm) - 1
    data = np.squeeze(readin["canopy_leaf_area_index"][tidx, :, :])

    yt = yt[index1[0] - 1 : index1[-1] + 2]
    xt = xt[index2[0] - 1 : index2[-1] + 2]
    data = data[index1[0] - 1 : index1[-1] + 2, index2[0] - 1 : index2[-1] + 2]

    xt_grid, yt_grid = np.meshgrid(xt, yt)
    data_grid = mapping(
        LAT, LON, data.ravel(), yt_grid.ravel(), xt_grid.ravel(), "nearest", np.nan
    )
    data_grid[data_grid < 0] = 0

    data_grid = PAI_cal(data_grid, INPUT[:, :, INPUTLIST.index("ast")])

    bai = BAI_reader(lat_lim, lon_lim, LAT, LON) * 10
    data_grid[INPUT[:, :, INPUTLIST.index("ast")] == 13] = (
        data_grid[INPUT[:, :, INPUTLIST.index("ast")] == 13]
        + bai[INPUT[:, :, INPUTLIST.index("ast")] == 13]
    )
    # data_grid = data_grid + (bai * 10)

    INPUT[:, :, INPUTLIST.index("pai")] = np.copy(data_grid)
    del [
        filename,
        readin,
        yt,
        xt,
        yt_grid,
        xt_grid,
        index1,
        index2,
        data,
        data_grid,
        bai,
    ]

    # fuel
    f_fuel = f_output[:20] + "/fuel.restart"

    if forecast_hour == 0:
        fuel = np.ones(LAT.shape)

        if init_fuel != 1:  # assign prefire fuel fraction
            readin = Dataset(f_input)
            time = readin["time"][0]
            curfire = readin["frp"][0, :, :]
            mask = curfire != 0
            readin.close()

            fuel[mask] = init_fuel
            del [readin, curfire, mask]

    else:
        fuel = np.loadtxt(f_fuel + ".f" + ("%02d" % (forecast_hour - 1)) + ".txt")

        if fuel_consume:
            readin = Dataset(f_input)
            prefire = readin["grid_predic"][0, :, :]
            mask = prefire != 0
            readin.close()

            cr = np.ones(LAT.shape)
            cr[mask] = cr_cal_by_frp(
                INPUT[:, :, INPUTLIST.index("ast")][mask], prefire[mask], 3000
            )
            fuel = fuel * cr
            del [readin, prefire, mask, cr]
            print("Fuel fraction updated.")
        else:
            print(f"Fixed fuel fraction used: {init_fuel}.")

    np.savetxt(f_fuel + ".f" + ("%02d" % forecast_hour) + ".txt", fuel, comments="")
    INPUT[:, :, INPUTLIST.index("fuel")] = np.copy(fuel)
    INPUT[:, :, INPUTLIST.index("ch")] = INPUT[:, :, INPUTLIST.index("ch")] * fuel
    INPUT[:, :, INPUTLIST.index("pai")] = INPUT[:, :, INPUTLIST.index("pai")] * fuel
    del [f_fuel, fuel]

    # t2m
    filename = file_finder("t2m", dd, initial_hour, forecast_hour, fcst_freq=time_freq)

    if not os.path.isfile(filename):
        print("---- Incorrect input file:", "t2m", "Terminated!")
        print(filename)
        return 1

    if mete_source == 0:
        yt, xt, data = hrrr_reader("t2m", filename)
    elif mete_source == 1:
        yt, xt, data = rap_reader("t2m", filename)
    elif mete_source == 2:
        yt, xt, data = gfs_reader("t2m", filename)
    elif mete_source == 3:
        yt, xt, data = wofs_reader("t2m", filename)

    data_grid = mapping(
        LAT, LON, data.ravel(), yt.ravel(), xt.ravel(), "linear", np.nan
    )
    data_grid[data_grid < 0] = np.nan

    INPUT[:, :, INPUTLIST.index("t2m")] = np.copy(data_grid)

    del [filename, yt, xt, data, data_grid]

    # vpd
    filename = file_finder("vpd", dd, initial_hour, forecast_hour, fcst_freq=time_freq)

    if os.path.isfile(filename) is False:
        print("---- Incorrect input file:", "vpd", "Terminated!")
        print(filename)
        return 1

    if mete_source == 0:
        yt, xt, data = hrrr_reader("vpd", filename)
    elif mete_source == 1:
        yt, xt, data = rap_reader("vpd", filename)
    elif mete_source == 2:
        yt, xt, data = gfs_reader("vpd", filename)
    elif mete_source == 3:
        yt, xt, data = wofs_reader("vpd", filename)

    data_grid = mapping(
        LAT, LON, data.ravel(), yt.ravel(), xt.ravel(), "linear", np.nan
    )
    data_grid[data_grid < 0] = np.nan

    INPUT[:, :, INPUTLIST.index("vpd")] = np.copy(data_grid)

    del [filename, yt, xt, data, data_grid]

    # prate
    filename = file_finder(
        "prate", dd, initial_hour, forecast_hour, fcst_freq=time_freq
    )

    if os.path.isfile(filename) is False:
        print("---- Incorrect input file:", "prate", "Terminated!")
        print(filename)
        return 1

    if mete_source == 0:
        yt, xt, data = hrrr_reader("prate", filename)
    elif mete_source == 1:
        yt, xt, data = rap_reader("prate", filename)
    elif mete_source == 2:
        yt, xt, data = gfs_reader("prate", filename)
    elif mete_source == 3:
        yt, xt, data = wofs_reader("prate", filename, fcst_hour=forecast_hour)

    data_grid = mapping(
        LAT, LON, data.ravel(), yt.ravel(), xt.ravel(), "linear", np.nan
    )
    data_grid[data_grid < 0] = np.nan

    INPUT[:, :, INPUTLIST.index("prate")] = np.copy(data_grid)

    del [filename, yt, xt, data, data_grid]

    # ws, wd
    filename = file_finder("ws", dd, initial_hour, forecast_hour, fcst_freq=time_freq)

    if os.path.isfile(filename) is False:
        print("---- Incorrect input file:", "ws, wd", "Terminated!")
        print(filename)
        return 1

    if mete_source == 0:
        yt, xt, u, v = hrrr_reader("ws", filename)
    elif mete_source == 1:
        yt, xt, u, v = rap_reader("ws", filename)
    elif mete_source == 2:
        yt, xt, u, v = gfs_reader("ws", filename)
    elif mete_source == 3:
        yt, xt, data = wofs_reader("ws", filename)

    u_new, v_new = wind_conversion(yt, u, v)

    u_grid = mapping(LAT, LON, u_new.ravel(), yt.ravel(), xt.ravel(), "linear", np.nan)
    v_grid = mapping(LAT, LON, v_new.ravel(), yt.ravel(), xt.ravel(), "linear", np.nan)

    # save for canopy wind calculation
    UGRID = np.copy(u_grid)
    VGRID = np.copy(v_grid)

    # ws, lat x lon
    ws_grid = np.sqrt(u_grid**2 + v_grid**2)

    # wd, lat x lon
    u_grid = units.Quantity(u_grid, "m/s")
    v_grid = units.Quantity(v_grid, "m/s")
    wd_grid = wind_direction(u_grid, v_grid, convention="from")

    INPUT[:, :, INPUTLIST.index("ws")] = np.copy(ws_grid)
    INPUT[:, :, INPUTLIST.index("wd")] = np.copy(wd_grid)
    del [filename, yt, xt, u, v, u_new, v_new, u_grid, v_grid, ws_grid, wd_grid]

    # soilw
    filename = file_finder(
        "soilw", dd, initial_hour, forecast_hour, fcst_freq=time_freq
    )

    if not os.path.isfile(filename):
        print("----Incorrect input file:", "soilw", "Terminated!")
        print(filename)
        exit()

    if mete_source == 0:
        yt, xt, data = hrrr_reader("soilw", filename)
    elif mete_source == 1:
        yt, xt, data = rap_reader("soilw", filename)
    elif mete_source == 2:
        yt, xt, data = gfs_reader("soilw", filename)
    elif mete_source == 3:
        yt, xt, data = wofs_reader("soilw", filename)

    data_grid = mapping(
        LAT, LON, data.ravel(), yt.ravel(), xt.ravel(), "linear", np.nan
    )
    data_grid[data_grid > 1] = 1
    data_grid[data_grid < 0] = 0

    INPUT[:, :, INPUTLIST.index("soilw")] = np.copy(data_grid)
    del [filename, yt, xt, data, data_grid]

    print("Start framing... Original data:", INPUT.shape)

    # ---- Fire Framing ----
    XX = LAT.shape[0]
    YY = LAT.shape[1]

    total = 0
    skip = 0

    # fire mask
    MASK = np.zeros(LAT.shape)
    MASK[FRP != 0] = 1

    # fire location
    lw, num = ndimage.label(MASK)
    lw = lw.astype(float)
    lw[lw == 0] = np.nan

    # fire frame
    INPUTFRAME = LATFRAME = LONFRAME = None
    for j in np.arange(1, num + 1, 1):
        total = total + 1
        index = np.argwhere(lw == j)

        if index.shape[0] == 1:
            loc = np.squeeze(index)
        else:
            xx = np.mean(LAT[index[:, 0], index[:, 1]])
            yy = np.mean(LON[index[:, 0], index[:, 1]])
            dis = (LAT - xx) ** 2 + (LON - yy) ** 2
            loc = np.squeeze(np.argwhere(dis == np.min(dis)))
            del [xx, yy, dis]

        if loc.size > 2:
            loc = loc[0, :]

        if (
            (loc[0] - fsize < 0)
            or (loc[0] + fsize + 1 > XX)
            or (loc[1] - fsize < 0)
            or (loc[1] + fsize + 1 > YY)
        ):
            skip = skip + 1
            continue

        X_fire = INPUT[
            loc[0] - fsize : loc[0] + fsize + 1, loc[1] - fsize : loc[1] + fsize + 1, :
        ]
        X_lat = LAT[
            loc[0] - fsize : loc[0] + fsize + 1, loc[1] - fsize : loc[1] + fsize + 1
        ]
        X_lon = LON[
            loc[0] - fsize : loc[0] + fsize + 1, loc[1] - fsize : loc[1] + fsize + 1
        ]

        if j == 1:
            INPUTFRAME = np.expand_dims(X_fire, axis=0)
            LATFRAME = np.expand_dims(X_lat, axis=0)
            LONFRAME = np.expand_dims(X_lon, axis=0)
        else:
            INPUTFRAME = np.append(INPUTFRAME, np.expand_dims(X_fire, axis=0), axis=0)
            LATFRAME = np.append(LATFRAME, np.expand_dims(X_lat, axis=0), axis=0)
            LONFRAME = np.append(LONFRAME, np.expand_dims(X_lon, axis=0), axis=0)

        del [index, loc, X_fire, X_lat, X_lon]
    del [lw, num, INPUT, MASK]

    if ("INPUTFRAME" not in locals()) or (INPUTFRAME.shape[0] == 0):
        print("---- " + tt + " no available frames.")
        return 2

    # remove frames with NaN
    index = []
    for i in np.arange(NN):
        X = np.sum(INPUTFRAME[:, :, :, i], axis=(1, 2))
        index = np.append(index, np.squeeze(np.argwhere(np.isnan(X))))
        del X

    index = index.astype(int)
    index = np.unique(index)

    if index.size != 0:
        INPUTFRAME = np.delete(INPUTFRAME, index, axis=0)
        LATFRAME = np.delete(LATFRAME, index, axis=0)
        LONFRAME = np.delete(LONFRAME, index, axis=0)
    nan = index.size
    del index

    # remove isolated small fires
    # fire_map = np.copy(INPUTFRAME[:, :, :, 0])
    # fire_map[fire_map != 0] = 1
    # fire_count = np.sum(fire_map, axis=(1, 2))
    # fire_max = np.max(INPUTFRAME[:, :, :, 0], axis=(1, 2))
    # index = np.squeeze(np.argwhere((fire_count == 1) & (fire_max < frp_thres)))

    # if index.size != 0:
    #    INPUTFRAME = np.delete(INPUTFRAME, index, axis=0)
    #    LATFRAME = np.delete(LATFRAME, index, axis=0)
    #    LONFRAME = np.delete(LONFRAME, index, axis=0)
    # small = index.size
    # del [fire_map, fire_count, fire_max, index]

    # remove frames with water bodies (AST=17)
    # index = np.argwhere(INPUTFRAME[:, :, :, INPUTLIST.index("ast")] == 17)[:, 0]
    # index = np.unique(index)

    # if index.size != 0:
    #    INPUTFRAME = np.delete(INPUTFRAME, index, axis=0)
    #    LATFRAME = np.delete(LATFRAME, index, axis=0)
    #    LONFRAME = np.delete(LONFRAME, index, axis=0)
    # water = index.size
    # del index
    water = "Not applied"

    print("---- Fire framing complete!")

    if INPUTFRAME.shape[0] == 0:
        print("---- No available frames!")
        return 2
    else:
        print("All:", total, "Skip:", skip, "NaN:", nan, "Water", water)
        print("Final:", INPUTFRAME.shape)

    INPUTFRAME_ori = np.copy(INPUTFRAME)

    # ---- Scaling ----
    for X in ["frp", "prate"]:
        i = INPUTLIST.index(X)
        X1 = np.copy(INPUTFRAME[:, :, :, i])
        X1[X1 != 0] = np.log(X1[X1 != 0])
        INPUTFRAME[:, :, :, i] = np.copy(X1)
        del [i, X1]

    for X in ["vpd", "ws"]:
        i = INPUTLIST.index(X)
        X1 = np.copy(INPUTFRAME[:, :, :, i])
        X1 = np.sqrt(X1)
        INPUTFRAME[:, :, :, i] = np.copy(X1)
        del [i, X1]

    for X in ["ch", "pai"]:
        i = INPUTLIST.index(X)
        X1 = np.copy(INPUTFRAME[:, :, :, i])
        X1 = (X1) ** (1 / 3)
        INPUTFRAME[:, :, :, i] = np.copy(X1)
        del [i, X1]

    # ---- Normalization ----
    # normalization coef
    X = np.copy(INPUTLIST)
    X[X == "fuel"] = "ffrac"
    coef = pd.read_csv("./model/model_normalization_coef.txt", usecols=X)
    a = coef.loc[0].values
    b = coef.loc[1].values

    for i in np.arange(NN):
        X = np.copy(INPUTFRAME[:, :, :, i])
        X = (X - b[i]) / a[i]
        INPUTFRAME[:, :, :, i] = np.copy(X)
        del X

    # ---- Writing Model Input ----
    INPUTFRAME[np.isnan(INPUTFRAME)] = -999

    f = Dataset(f_output, "w")
    f.createDimension("time", 1)
    f.createDimension("flen", INPUTFRAME.shape[0])
    f.createDimension("xlen", INPUTFRAME.shape[1])
    f.createDimension("ylen", INPUTFRAME.shape[2])
    f.createDimension("xt", FRP.shape[0])
    f.createDimension("yt", FRP.shape[1])
    f.createDimension("num_input", NN)
    var_time = f.createVariable("time", str, ("time",))
    var_input0 = f.createVariable(
        "input_noscale", "float", ("flen", "xlen", "ylen", "num_input")
    )
    var_input = f.createVariable(
        "input", "float", ("flen", "xlen", "ylen", "num_input")
    )
    var_lat = f.createVariable("frame_lat", "float", ("flen", "xlen", "ylen"))
    var_lon = f.createVariable("frame_lon", "float", ("flen", "xlen", "ylen"))
    var_ygrid = f.createVariable("grid_lat", "float", ("time", "xt", "yt"))
    var_xgrid = f.createVariable("grid_lon", "float", ("time", "xt", "yt"))
    var_frp = f.createVariable("grid_frp", "float", ("time", "xt", "yt"))
    var_list = f.createVariable("INPUTLIST", str, ("num_input",))

    var_time[:] = np.array(time).astype(str)
    var_input0[:] = INPUTFRAME_ori
    var_input[:] = INPUTFRAME
    var_lat[:] = LATFRAME
    var_lon[:] = LONFRAME
    var_ygrid[:] = LAT
    var_xgrid[:] = LON
    var_frp[:] = np.expand_dims(FRP, axis=0)
    var_list[:] = np.array(INPUTLIST).astype(str)
    f.close()

    return 0
