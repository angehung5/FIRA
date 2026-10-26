import numpy as np


def cr_cal_by_fraction(vtype, burnt):
    # resuming fuel = ratio of fuel fractions
    from fuel_weight import fw_rothermal

    w = fw_rothermal(vtype)
    tf = w / 0.8514
    f0 = np.exp(-1 * (burnt - 1) / tf)
    f1 = np.exp(-1 * burnt / tf)
    cr = f1 / f0
    cr[(vtype == 11) | (vtype == 15) | (vtype == 16) | (vtype == 17)] = 0
    return cr


def cr_cal_by_frp(vtype, frp, dx):
    intensity = frp / (dx**2)  # fire intensity(MW/m2)
    a = np.zeros(vtype.shape)  # 0.368/fl based on van Leeuwen et al. (2014)

    # needleleaf forest
    a[(vtype == 1) | (vtype == 3)] = 1.21732773e-01

    # broadleaf forest
    a[(vtype == 2) | (vtype == 4)] = 2.37522091e-02

    # mixed forest/urban
    a[(vtype == 5) | (vtype == 13)] = 1.45320768e-01

    # shrubs
    a[(vtype == 6) | (vtype == 7)] = 1.70857143e-01

    # grass/crop
    a[(vtype == 8) | (vtype == 9) | (vtype == 10) | (vtype == 12) | (vtype == 14)] = (
        6.00845873e-01
    )

    cr = a * intensity  # consumed fraction (%/s)
    cr = 1 - (cr * 3600)  # resuming fuel (%/h)
    cr[(vtype == 11) | (vtype == 15) | (vtype == 16) | (vtype == 17)] = 0
    cr[cr < 0] = 0.1
    cr[frp == 0] = 1
    return cr
