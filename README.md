# Fire Intensity and spRead forecAst (FIRA)

Repository for a fire spread forecast model, generating hourly fire radiative power (FRP) prediction for air quality forecasting applications.
Please check `requirement.txt` for required python libraries.

## Log
Sep 2024: Repository initialized; FIRAv1
Oct 2026: FIRAv2
    - Update input variables and model configurations
    - Add fuel parameters (fai, ff)

## Components

Trained machine learning models are available in `model/fira_models.zip`

| **Source Code (Python Script)**   | **Script Description**                                                                                             |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| `fire_model`                      | Main driver of FIRA                                                                                                |
| `frp_preprocessor`                | Data preprocessor for initial FRP data                                                                             |
| `fire_inputgen`                   | Model input generator, including data re-gridding, fire frame selection and filtering, scaling, and normalization  |
| `fire_forecast`                   | Driver for fire spread prediction model                                                                            |
| `fire_intensity`                  | Driver for fire intensity prediction model                                                                         |
| `fire_mapgen`                     | Fire mapper for grided FRP prediction generation                                                                   |
| `fire_evaluation`                 | Visualization tool for grided FRP prediction                                                                       |
| `mete_reader`                     | Reader toolkit for meteorological fields                                                                           |
| `fuel_consumption_cal`            | Estimation tool for fuel consumption                                                                               |
| `fuel_weight`                     | Estimation tool for fuel weight                                                                                    |

## Namelist settings

| **Namelist Option** | **Option Description**                                                 |
| ------------------- | ---------------------------------------------------------------------- |
| Filename specification                                                                       |
| `frp_input`         | Filename of FRP input files                                            |
| `model_input`       | Filename of FIRA input files                                           |
| `model_output`      | Filename of FIRA output files                                          |
| Data source option                                                                           |
| `frp_source`        | Option of the source of initial FRP data (0: RAVE, 1: ABI L2)          |
| `mete_source`       | Option of the source of meteorological data (0: HRRR, 1: RAP, 2: GFS, 3: WoFS)       |
| Simulation time settings                                                                     |
| `time_start`        | Simulation start time in UTC (YYYYMMDDHHMM)                            |
| `time_end`          | Simulation end time in UTC (YYYYMMDDHHMM)                              |
| `time_freq`         | Temporal frequency of simulation, unit depends on input                |
| Simulation domain settings                                                                   |
| `lat_lower_left`    | Latitude of the lower left point of model domain (degree)              |
| `lat_upper_right`   | Latitude of the upper right point of model domain (degree)             |
| `lon_lower_left`    | Longitude of the lower left point of model domain (degree)             |
| `lon_upper_right`   | Longitude of the upper right point of model domain (degree)            |
| Model options                                                                                |
| `opt_frpgen`        | Option for FRP preprocessor (0: off, 1: on)                            |
| `opt_inputgen`      | Option for input generator (0: off, 1: on)                             |
| `opt_forecast`      | Option for fire spread prediction model (0: off, 1: on)                |
| `opt_intensity`     | Option for fire intensity prediction model (0: off, 1: on)             |
| `opt_mapgen`        | Option for fire mapper (0: off, 1: on)                                 |
| `opt_evaluation`    | Option for result visualization (0: off, 1: on)                        |
| Spread simulation settings                                                                   |
| `min_spread_thres`  | Threshold of possible fire occurrence (default=0.2). Predictions equal to or higher than the threshold are identified as active fires                                      |
| `fuel_consume`      | Option for dynamic fuel consumption (0: off, 1: on)                    |
| `fuel_max`          | Maximum fuel fraction (default=1)                                      |
| `fuel_min`          | Maximum fuel fraction (default=0)                                      |
| Manual/Fake fire settings                                                                    |
| `man_fire`          | Option for adding manual fire (0: off, 1: on)                          |
| `man_lat`           | Latitude of manual fires, separated by `,` (degree)                    |
| `man_lon`           | Longitude of manual fires, separated by `,` (degree)                   |
| `man_lvl`           | Fire intensity (FRP) of manual fires, separated by `,` (MW)            |
| Path for required input files                                                                |
| `path_frp`          | Path for initial FRP files                                             |
| `path_elv`          | Path for terrain elevation files                                       |
| `path_ast`          | Path for surface type files                                            |
| `path_fh`           | Path for forest height files                                           |
| `path_vhi`          | Path for vegetation health index files                                 |
| `path_mete`         | Path for meteorological data files                                     |

## Inputs

1. Filename format: `[input name].[start time].f[simuluation time].nc`
   `input name` and `start time` can be specified in namelist.

2. Dynamic fuel map: `fuel.restart.f[simuluation time].txt`
   They are intermediate files generated during the simulation.

3. Input netcdf file components:

| **Variable Name**   | **Variable Description**                                                           |
| ------------------- | ---------------------------------------------------------------------------------- |
| `time`              | Time in UTC                                                                        |
| `INPUTLIST`         | List of input variables                                                            |
| `input_noscale`     | Input variables per fire frames before scaling and normalization (units vary)      |
| `input`             | Input variables per fire frames after scaling and normalization (dimensionless)    |
| `frame_lat`         | Latitude of fire frames (degree)                                                   |
| `frame_lon`         | Longitude of fire frames (degree)                                                  |

3. List of input variables:

| **Variable Name** | **Variable Description**                  | **Data Source**                                                                 |
| ----------------- | ----------------------------------------- | ------------------------------------------------------------------------------- |
| Fire characteristics                                                                                                                            |
| `frp`             | Fire radiative power (MW)                 | [RAVE](https://sites.google.com/view/rave-emission/)                            |
| Fuel/surface characteristics                                                                                                                    |
| `ch`              | Canopy height (m)                         | GEDI ([Lang et al.](https://langnico.github.io/globalcanopyheight/))            |
| `st`              | Surface type (dimensionless)              | [VIIRS AST](https://www.star.nesdis.noaa.gov/jpss/st.php)                       |
| `fai`             | Fuel area index (m2/m2)                   | [VIIRS LAI](https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.ncdc:C01472); [MS Global building footprint](https://github.com/microsoft/globalmlbuildingfootprints) |
| `ff`              | Fuel fraction (dimensionless)             | N/A                                                                             |
| Meteorological conditions                                                                                                                       |
| `t2m`             | 2-meter temperature                       | [HRRR](https://rapidrefresh.noaa.gov/hrrr/)/[University of Utah](https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/hrrr_download.cgi)   |
| `vpd`             | 2-meter Vapor pressure deficit            | [HRRR](https://rapidrefresh.noaa.gov/hrrr/)/[University of Utah](https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/hrrr_download.cgi); Estimated based on [Chiodi et al., 2021](https://doi.org/10.1029/2021GL092830)   |
| `prate`           | Precipitation rate (mm h-1)               | [HRRR](https://rapidrefresh.noaa.gov/hrrr/)/[University of Utah](https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/hrrr_download.cgi)   |
| `wd`              | 10-meter wind direction (degree)          | [HRRR](https://rapidrefresh.noaa.gov/hrrr/)/[University of Utah](https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/hrrr_download.cgi)   |
| `ws`              | 10-meter wind speed (m s-1)               | [HRRR](https://rapidrefresh.noaa.gov/hrrr/)/[University of Utah](https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/hrrr_download.cgi)   |
| `soilw`           | Soil mositure (m3/m3)                     | [HRRR](https://rapidrefresh.noaa.gov/hrrr/)/[University of Utah](https://home.chpc.utah.edu/~u0553130/Brian_Blaylock/cgi-bin/hrrr_download.cgi)   |

## Outputs

1. Filename format: `[output name].[start time].f[forecast hour].nc`
   `output name` and `start time` can be specified in namelist.

2. Output netCDF file components:

| **Variable Name**   | **Variable Description**                                               |
| ------------------- | ---------------------------------------------------------------------- |
| `time`              | Time of FRP prediction in UTC                                          |
| `INPUTLIST`         | List of input variables                                                |
| `frame_predic_ori`  | Original fire spread prediction per fire frames (dimensionless)        |
| `frame_predic_post` | Post-processed fire spread prediction per fire frames (dimensionless)  |
| `frame_predic_frp`  | Fire intensity (FRP) prediction per fire frames (MW)                   |
| `frame_lat`         | Latitude of fire frames (degree)                                       |
| `frame_lon`         | Longitude of fire frames (degree)                                      |
| `grid_predic`       | Gridded FRP prediction (MW)                                            |
| `grid_lat`          | Latitude of gridded FRP prediction (degree)                            |
| `grid_lon`          | Longitude of gridded FRP prediction (degree)                           |

## Citation

Hung et al. (2025). Fire Intensity and spRead forecAst (FIRA): A machine learning based fire spread prediction model for air quality forecasting application. GeoHealth, 9, e2024GH001253. https://doi.org/10.1029/2024GH001253.

## Disclaimer

This repository is a scientific product and is not official communication of the National Oceanic and Atmospheric Administration, or the United States Department of Commerce. All NOAA GitHub project code is provided on an "as is" basis and the user assumes responsibility for its use. Any claims against the Department of Commerce or Department of Commerce bureaus stemming from the use of this GitHub project will be governed by all applicable Federal law. Any reference to specific commercial products, processes, or services by service mark, trademark, manufacturer, or otherwise, does not constitute or imply their endorsement, recommendation or favoring by the Department of Commerce. The Department of Commerce seal and logo, or the seal and logo of a DOC bureau, shall not be used in any manner to imply endorsement of any commercial product or activity by DOC or the United States Government.
