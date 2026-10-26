"""
Author: whung

Imperial units: ft, lb, Btu, min

Input parameters:
    vtype - Fuel type (Dimensionless), classified into 3 types based on Andrews (2018)

Prescribed parameters:
    W0    - Total fuel loading (lb/ft2)
    rho   - Fuel bulk density (lb/ft3)
    csav  - Characteristic surface-area-to-volume-ratio (ft2/ft3)
    delta - Fuel bed depth (ft)
    beta  - Relative packing ratio (Dimensionless)

Calculated parameters:
    gamma_max - Maximum reaction velocity (min-1)
    beta_op   - Optimum packing ratio (Dimensionless)

Output Parameters:
    gamma     - Optimum reaction velocity (min-1)

Key references:
    Rothermel (1972) https://www.fs.usda.gov/rm/pubs_int/int_rp115.pdf
    Andrews (2018) https://www.fs.usda.gov/rm/pubs_series/rmrs/gtr/rmrs_gtr371.pdf
"""

import warnings

import numpy as np

warnings.simplefilter(action="ignore")


def fw_rothermal(vtype):
    # Bulk particle density = 32 lb/ft3

    W0 = np.ones(vtype.shape)
    csav = np.ones(vtype.shape)
    delta = np.ones(vtype.shape)
    Mx = np.ones(vtype.shape)

    # needleleaf forest
    W0[(vtype == 1) | (vtype == 3)] = (
        (1.5 + 1.0 + 2.5)
        + (2.9 + 0.41 + 0.15)
        + (3.0 + 2.0 + 5.0 + 2.0)
        + (1.0 + 2.2 + 3.6)
        + (0.5 + 2.2 + 2.8)
        + (1.15 + 2.5 + 4.4)
        + (5.8 + 1.4 + 1.1)
        + (0.5 + 1.5 + 4.2)
        + (0.3 + 1.4 + 8.1)
    ) / 9.0
    csav[(vtype == 1) | (vtype == 3)] = (
        1889 + 2484 + 1764 + 1716 + 1532 + 1713 + 1770 + 1568 + 1229
    ) / 9.0
    delta[(vtype == 1) | (vtype == 3)] = (
        0.2 + 0.2 + 1 + 0.2 + 0.3 + 0.6 + 0.3 + 0.4 + 0.4
    ) / 9.0
    Mx[(vtype == 1) | (vtype == 3)] = (30 + 25 + 25 + 30 + 20 + 25 + 35 + 25 + 25) / 9.0

    # broadleaf forest
    W0[(vtype == 2) | (vtype == 4)] = (
        (1.1 + 1.9 + 1.5 + 0.37)
        + (3.0 + 2.0 + 5.0 + 2.0)
        + (1.4 + 2.3 + 2.2)
        + (2.4 + 1.2 + 1.2)
        + (6.65 + 3.3 + 4.15)
        + (0.5 + 1.5 + 4.2)
        + (0.3 + 1.4 + 8.1)
    ) / 7.0
    csav[(vtype == 2) | (vtype == 4)] = (
        1552 + 1764 + 1806 + 1936 + 1733 + 1568 + 1229
    ) / 7.0
    delta[(vtype == 2) | (vtype == 4)] = (2.5 + 1.0 + 0.2 + 0.3 + 0.6 + 0.4 + 0.4) / 7.0
    Mx[(vtype == 2) | (vtype == 4)] = (40 + 25 + 25 + 25 + 35 + 25 + 35) / 7.0

    # mixed forest/urban
    W0[(vtype == 5) | (vtype == 13)] = (
        (1.5 + 1.0 + 2.5)
        + (2.9 + 0.41 + 0.15)
        + (3.0 + 2.0 + 5.0 + 2.0)
        + (1.0 + 2.2 + 3.6)
        + (0.5 + 2.2 + 2.8)
        + (1.15 + 2.5 + 4.4)
        + (5.8 + 1.4 + 1.1)
        + (1.1 + 1.9 + 1.5 + 0.37)
        + (3.0 + 2.0 + 5.0 + 2.0)
        + (1.4 + 2.3 + 2.2)
        + (2.4 + 1.2 + 1.2)
        + (6.65 + 3.3 + 4.15)
        + (0.5 + 1.5 + 4.2)
        + (0.3 + 1.4 + 8.1)
    ) / 14.0
    csav[(vtype == 5) | (vtype == 13)] = (
        1889
        + 2484
        + 1764
        + 1716
        + 1532
        + 1713
        + 1770
        + 1552
        + 1764
        + 1806
        + 1936
        + 1733
        + 1568
        + 1229
    ) / 14.0
    delta[(vtype == 5) | (vtype == 13)] = (
        0.2 + 0.2 + 1 + 0.2 + 0.3 + 0.6 + 0.3 + 2.5 + 1.0 + 0.2 + 0.3 + 0.6 + 0.4 + 0.4
    ) / 14.0
    Mx[(vtype == 5) | (vtype == 13)] = (
        30 + 25 + 25 + 30 + 20 + 25 + 35 + 40 + 25 + 25 + 25 + 35 + 25 + 25
    ) / 14.0

    # shrubs
    W0[(vtype == 6) | (vtype == 7)] = (
        (5.0 + 4.0 + 2.0 + 5.0)
        + (1.0 + 0.5 + 2.0)
        + (1.5 + 2.5 + 2.0)
        + (0.2 + 0.5 + 0.65)
        + (0.5 + 0.5 + 0.6 + 1.0)
        + (0.3 + 0.25 + 1.45 + 1.25)
        + (1.9 + 0.3 + 0.1 + 3.4 + 7.1)
        + (0.25 + 0.25 + 0.15 + 1.3)
        + (1.35 + 2.4 + 0.75 + 3.85)
        + (0.45 + 3.0 + 6.2)
        + (0.85 + 1.15 + 0.2 + 2.55)
        + (3.6 + 2.1 + 2.9)
        + (2.9 + 1.45 + 1.4)
        + (3.5 + 5.3 + 2.2 + 3.4)
        + (2.05 + 3.4 + 0.85 + 4.35)
        + (4.5 + 2.45 + 1.55 + 7.0)
        + (0.2 + 0.9 + 1.5 + 0.2 + 0.9)
    ) / 17.0
    csav[(vtype == 6) | (vtype == 7)] = (
        1739
        + 1683
        + 1564
        + 1832
        + 1827
        + 1614
        + 1631
        + 1674
        + 1672
        + 1371
        + 1682
        + 1252
        + 1144
        + 1233
        + 1386
        + 1378
        + 1606
    ) / 17.0
    delta[(vtype == 6) | (vtype == 7)] = (
        6.0
        + 2.0
        + 2.5
        + 0.9
        + 1.5
        + 1.8
        + 2.1
        + 1.0
        + 1.0
        + 2.4
        + 3.0
        + 6.0
        + 2.0
        + 6.0
        + 3.0
        + 4.4
        + 0.6
    ) / 17.0
    Mx[(vtype == 6) | (vtype == 7)] = (
        20
        + 20
        + 25
        + 15
        + 15
        + 40
        + 40
        + 15
        + 15
        + 40
        + 30
        + 15
        + 30
        + 15
        + 40
        + 40
        + 20
    ) / 17.0

    # grass/crop
    W0[(vtype == 8) | (vtype == 9) | (vtype == 10) | (vtype == 12) | (vtype == 14)] = (
        0.74
        + (2.0 + 1.0 + 0.5 + 0.5)
        + 3.0
        + (0.1 + 0.3)
        + (0.1 + 1.0)
        + (0.1 + 0.4 + 1.5)
        + (0.25 + 1.9)
        + (0.4 + 2.5)
        + (0.1 + 3.4)
        + (1.0 + 5.4)
        + (0.5 + 7.3)
        + (1.0 + 1.0 + 9.0)
    ) / 12.0
    csav[
        (vtype == 8) | (vtype == 9) | (vtype == 10) | (vtype == 12) | (vtype == 14)
    ] = (
        3500
        + 2784
        + 1500
        + 2054
        + 1820
        + 1290
        + 1826
        + 1631
        + 2006
        + 1834
        + 1302
        + 1812
    ) / 12.0
    delta[
        (vtype == 8) | (vtype == 9) | (vtype == 10) | (vtype == 12) | (vtype == 14)
    ] = (1.0 + 1.0 + 2.5 + 0.4 + 1.0 + 2.0 + 2.0 + 1.5 + 1.5 + 3.0 + 4.0 + 5.0) / 12.0
    Mx[(vtype == 8) | (vtype == 9) | (vtype == 10) | (vtype == 12) | (vtype == 14)] = (
        12 + 15 + 25 + 15 + 15 + 30 + 15 + 40 + 40 + 15 + 30 + 40
    ) / 12.0

    W0 = W0 * (2000 / 43560)  # tons/acre -> lb/ft2
    rho = W0 / delta
    beta = rho / 32.0

    gamma_max = (csav ** (1.5)) * (495 + (0.0594 * (csav**1.5))) ** (-1)
    beta_op = 3.348 * (csav ** (-0.8189))
    A = 133 * (csav ** (-0.7913))
    B = beta / beta_op

    gamma = gamma_max * (B**A) * np.exp(A * (1 - B))
    fw = 1 / (gamma / 60)  # min -> h
    return fw


def fw_wrffire(vtype):
    # Classify Anderson 13 surface fuel models to VIIRS vtype
    # FUEL MODEL 1: Short grass (1 ft), fw = 7min
    # FUEL MODEL 2: Timber (grass and understory), fw = 7min
    # FUEL MODEL 3: Tall grass (2.5 ft), fw = 7min
    # --- Shrub-dominated fuel models
    # FUEL MODEL 4: Chaparral (6 ft), fw = 180min
    # FUEL MODEL 5: Brush (2 ft), fw = 100min
    # FUEL MODEL 6: Dormant brush, hardwood slash, fw = 100min
    # FUEL MODEL 7: Southern rough, fw = 100min
    # --- Forest litter-dominated fuel models
    # FUEL MODEL 8: Closed timber litter, fw = 900min
    # FUEL MODEL 9: Hardwood litter, fw = 900min
    # FUEL MODEL 10: Timber (litter + understory), fw = 900min
    # --- Logging debris-dominated fuel models
    # FUEL MODEL 11: Light logging slash, fw = 900min
    # FUEL MODEL 12: Medium logging slash, fw = 900min
    # FUEL MODEL 13: Heavy logging slash, fw = 7min
    # https://github.com/NOAA-EMC/fire_behavior/blob/05cad173feeb598431e3ef5f17c2df6562c8d101/physics/fuel_anderson_mod.F90

    fw = np.zeros(vtype.shape)
    fw[(vtype <= 5) | (vtype == 13)] = (900 + 900 + 900 + 900 + 900 + 7) / 6
    fw[(vtype == 6) | (vtype == 7)] = (180 + 100 + 100 + 100) / 4
    fw[(vtype == 8) | (vtype == 9) | (vtype == 10) | (vtype == 12) | (vtype == 14)] = (
        7 + 7 + 7
    ) / 3

    return fw / 60  # min -> h
