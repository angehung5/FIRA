import numpy as np
import xarray as xr


def hrrr_reader(var, filename):
    if var == "t2m":
        readin = xr.open_dataset(
            filename,
            engine="cfgrib",
            filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2},
        )
        yt = readin["latitude"].data
        xt = readin["longitude"].data
        data = readin["t2m"].data
        return yt, xt, data

    elif var == "vpd":
        readin = xr.open_dataset(
            filename,
            engine="cfgrib",
            filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2},
        )
        yt = readin["latitude"].data
        xt = readin["longitude"].data
        t = readin["t2m"].data - 273.15
        td = readin["d2m"].data - 273.15

        # calculation based on Chiodi et al. 2021
        # https://doi. org/10.1029/2021GL092830
        data = (6.109 * np.exp(17.625 * t / (t + 243.04))) - (
            6.109 * np.exp(17.625 * td / (td + 243.04))
        )
        data[data < 0] = 0
        return yt, xt, data

    elif var == "prate":
        readin = xr.open_dataset(
            filename,
            engine="cfgrib",
            filter_by_keys={"typeOfLevel": "surface", "stepType": "instant"},
        )
        yt = readin["latitude"].data
        xt = readin["longitude"].data
        data = readin["prate"].data
        data = data * 3600  # kg m-2 s-1 -> hourly accumulation
        return yt, xt, data

    elif (var == "ws") or (var == "wd"):
        readin = xr.open_dataset(
            filename,
            engine="cfgrib",
            filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 10},
        )
        yt = readin["latitude"].data
        xt = readin["longitude"].data
        u = readin["u10"].data
        v = readin["v10"].data
        return yt, xt, u, v

    elif var == "soilw":
        readin = xr.open_dataset(
            filename,
            engine="cfgrib",
            filter_by_keys={"typeOfLevel": "depthBelowLandLayer"},
        )
        yt = readin["latitude"].data
        xt = readin["longitude"].data
        data = np.trapz(
            readin["soilw"].data, readin["depthBelowLandLayer"].data, axis=0
        )
        return yt, xt, data


def rap_reader(var, filename):
    if var == "t2m":
        readin = xr.open_dataset(
            filename,
            engine="grib2io",
            filters={
                "productDefinitionTemplateNumber": 0,
                "typeOfFirstFixedSurface": 103,
                "valueOfFirstFixedSurface": 2.0,
            },
        )
        yt = readin["latitude"].data
        xt = readin["longitude"].data
        xt[xt < 0] = xt[xt < 0] + 360
        data = readin["TMP"].data
        return yt, xt, data

    elif var == "vpd":
        readin = xr.open_dataset(
            filename,
            engine="grib2io",
            filters={
                "productDefinitionTemplateNumber": 0,
                "typeOfFirstFixedSurface": 103,
                "valueOfFirstFixedSurface": 2.0,
            },
        )
        yt = readin["latitude"].data
        xt = readin["longitude"].data
        xt[xt < 0] = xt[xt < 0] + 360
        t = readin["TMP"].data - 273.15
        td = readin["DPT"].data - 273.15

        # calculation based on Chiodi et al. 2021
        # https://doi. org/10.1029/2021GL092830
        data = (6.109 * np.exp(17.625 * t / (t + 243.04))) - (
            6.109 * np.exp(17.625 * td / (td + 243.04))
        )
        data[data < 0] = 0
        return yt, xt, data

    elif var == "prate":
        readin = xr.open_dataset(
            filename,
            engine="grib2io",
            filters={
                "productDefinitionTemplateNumber": 0,
                "typeOfFirstFixedSurface": 1,
            },
        )
        yt = readin["latitude"].data
        xt = readin["longitude"].data
        xt[xt < 0] = xt[xt < 0] + 360
        data = readin["PRATE"].data
        data = data * 3600  # kg m-2 s-1 -> hourly accumulation
        return yt, xt, data

    elif (var == "ws") or (var == "wd"):
        readin = xr.open_dataset(
            filename,
            engine="grib2io",
            filters={
                "productDefinitionTemplateNumber": 0,
                "typeOfFirstFixedSurface": 103,
                "valueOfFirstFixedSurface": 10.0,
            },
        )
        yt = readin["latitude"].data
        xt = readin["longitude"].data
        xt[xt < 0] = xt[xt < 0] + 360
        u = readin["UGRD"].data
        v = readin["VGRD"].data
        return yt, xt, u, v

    elif var == "soilw":
        readin = xr.open_dataset(
            filename,
            engine="grib2io",
            filters={
                "productDefinitionTemplateNumber": 0,
                "typeOfFirstFixedSurface": 106,
            },
        )
        yt = readin["latitude"].data
        xt = readin["longitude"].data
        xt[xt < 0] = xt[xt < 0] + 360
        data = np.trapz(
            readin["SOILW"].data, readin["valueOfFirstFixedSurface"].data, axis=0
        )
        return yt, xt, data


def gfs_reader(var, filename):
    if var == "t2m":
        readin = xr.open_dataset(filename)
        xt, yt = np.meshgrid(readin["grid_xt"].data, readin["grid_yt"].data)
        data = readin["tmp2m"].data[0, :, :]
        return yt, xt, data

    elif var == "vpd":
        readin = xr.open_dataset(filename)
        xt, yt = np.meshgrid(readin["grid_xt"].data, readin["grid_yt"].data)
        t = readin["tmp2m"].data[0, :, :] - 273.15  # K -> C
        sh = readin["spfh2m"].data[0, :, :]  # kg/kg
        pr = readin["pressfc"].data[0, :, :] * 0.01  # Pa -> mb
        pr = pr * 0.1
        e = pr * sh / (0.622 + sh)
        es = 0.6112 * np.exp(17.67 * t / (t + 243.5))

        data = es - e
        data[data < 0] = 0
        return yt, xt, data

    elif var == "prate":
        readin = xr.open_dataset(filename)
        xt, yt = np.meshgrid(readin["grid_xt"].data, readin["grid_yt"].data)
        data = readin["prate_ave"].data[0, :, :]
        data = data * 3600  # kg m-2 s-1 -> hourly accumulation
        return yt, xt, data

    elif (var == "ws") or (var == "wd"):
        readin = xr.open_dataset(filename)
        xt, yt = np.meshgrid(readin["grid_xt"].data, readin["grid_yt"].data)
        u = readin["ugrd10m"].data[0, :, :]
        v = readin["vgrd10m"].data[0, :, :]
        return yt, xt, u, v

    elif var == "soilw":
        readin = xr.open_dataset(filename)
        xt, yt = np.meshgrid(readin["grid_xt"].data, readin["grid_yt"].data)

        soilw = np.append(readin["soilw1"].data, readin["soilw2"].data, axis=0)
        soilw = np.append(soilw, readin["soilw2"].data, axis=0)
        soilw = np.append(soilw, readin["soilw2"].data, axis=0)
        depth = np.array([5, 25, 70, 150]) / 100  # cm -> m

        data = np.trapz(soilw, depth, axis=0)
        return yt, xt, data


def wofs_reader(var, filename, fcst_hour=None):
    if var == "t2m":
        readin = xr.open_dataset(filename)
        xt = readin["xlon"].data
        yt = readin["xlat"].data
        xt[xt < 0] = xt[xt < 0] + 360
        data = np.mean(readin["t_2"].data, axis=0)
        data = ((data - 32) / 1.8) + 273.15  # F -> C -> K
        return yt, xt, data

    elif var == "vpd":
        readin = xr.open_dataset(filename)
        xt = readin["xlon"].data
        yt = readin["xlat"].data
        xt[xt < 0] = xt[xt < 0] + 360
        t = np.mean(readin["t_2"].data, axis=0)
        t = (t - 32) / 1.8  # F -> C
        sh = np.mean(readin["qv_2"].data, axis=0)  # kg/kg
        pr = np.mean(readin["p_sfc"].data, axis=0)  # hPa (mb)
        pr = pr * 0.1
        e = pr * sh / (0.622 + sh)
        es = 0.6112 * np.exp(17.67 * t / (t + 243.5))

        data = es - e
        data[data < 0] = 0
        return yt, xt, data

    elif var == "prate":
        readin = xr.open_dataset(filename[1])
        xt = readin["xlon"].data
        yt = readin["xlat"].data
        xt[xt < 0] = xt[xt < 0] + 360
        data1 = np.mean(readin["rain"].data, axis=0)  # mm, current time step
        readin.close()

        if fcst_hour == 0:  # initial time
            data0 = np.zeros(xt.shape)
        else:
            readin = xr.open_dataset(filename[0])
            data0 = np.mean(readin["rain"].data, axis=0)  # mm, previous time step
            readin.close()

        data = (
            data1 - data0
        ) / 1000  # mm -> m, total precipitation within single time window
        return yt, xt, data

    elif (var == "ws") or (var == "wd"):
        readin = xr.open_dataset(filename)
        xt = readin["xlon"].data
        yt = readin["xlat"].data
        xt[xt < 0] = xt[xt < 0] + 360
        u = np.mean(readin["u_10"].data, axis=0)
        v = np.mean(readin["v_10"].data, axis=0)
        return yt, xt, u, v

    elif var == "soilw":
        readin = xr.open_dataset(filename)
        xt = readin["xlon"].data
        yt = readin["xlat"].data
        xt[xt < 0] = xt[xt < 0] + 360

        soilw = readin["SMOIS"].data[0, :]
        depth = readin["ZS"].data[0, :]  # m

        data = np.trapz(soilw, depth, axis=0)
        return yt, xt, data
