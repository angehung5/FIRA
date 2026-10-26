"""
This is the main script to run the fire spread model.
"""

import os
import sys

import numpy as np
import pandas as pd

# ---- Settings ----
namelist = pd.read_csv("./input/namelist", header=None, delimiter="=")
namelist = namelist.rename(columns={0: "key", 1: "value"}).set_index("key")
namelist = namelist.to_dict()["value"]

# input/output files
frp_input = namelist["frp_input"]
model_input = namelist["model_input"]
model_output = namelist["model_output"]

# frp/mete source
frp_source = int(namelist["frp_source"])  # 0: rave
mete_source = int(namelist["mete_source"])  # 0: hrrr, 1: rap, 2: gfs

# forecast period
time_start = str(namelist["time_start"])  # yyyymmddHHMM
time_end = str(namelist["time_end"])  # yyyymmddHHMM
time_freq = int(namelist["time_freq"])  # unit depends on input
time = pd.date_range(start=time_start, end=time_end, freq=str(time_freq) + "H")
initial_time = time_start[8:10]

# domain
lat_lim = [float(namelist["lat_lower_left"]), float(namelist["lat_upper_right"])]
lon_lim = [float(namelist["lon_lower_left"]), float(namelist["lon_upper_right"])]

# function option
opt_frpgen = namelist["opt_frpgen"] in ["True", "true"]  # input generator option
opt_inputgen = namelist["opt_inputgen"] in ["True", "true"]  # input generator option
opt_forecast = namelist["opt_forecast"] in ["True", "true"]  # forecast model  option
opt_intensity = namelist["opt_intensity"] in ["True", "true"]  # intensity model option
opt_mapgen = namelist["opt_mapgen"] in ["True", "true"]  # FRP map generator option
opt_evaluation = namelist["opt_evaluation"] in [
    "True",
    "true",
]  # model evaluation option


TT = len(time)


if not os.path.exists("./input/" + time_start):
    os.makedirs("./input/" + time_start)
if not os.path.exists("./output/" + time_start):
    os.makedirs("./output/" + time_start)


# ---- Log saving ----
class StdoutLogger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


class StderrLogger:
    def __init__(self, filename):
        self.terminal = sys.stderr
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


f_log = (
    "./output/" + time_start + "/" + model_output + "." + time_start + "_" + time_end
)

sys.stdout = StdoutLogger(f_log + ".log")
sys.stderr = StderrLogger(f_log + ".err")


print("--------------------")
print("---- Fire spread model initializing...")
print(
    "---- Model cycle:",
    time_start,
    "-",
    time_end,
    ", freq=",
    time_freq,
    "H, cycle=",
    TT,
)
print("---- Model domain:", lat_lim, lon_lim)


f_frp = "./input/" + time_start + "/" + frp_input + "." + time_start + ".nc"


# ---- Creating initial FRP file ----
if opt_frpgen:
    if frp_source == 0:
        from frp_preprocessor import rave_preprocessor

        code = rave_preprocessor(frp_input, time_start, lat_lim, lon_lim)
        if code == 0:
            print("---- Initial FRP generated!")
        elif code == 1:
            print("No available or unknown initial FRP. Model Terminated!")
            exit()

    elif frp_source == 1:
        from frp_preprocessor import abi_preprocessor

        code = abi_preprocessor(frp_input, time_start, lat_lim, lon_lim)
        if code == 0:
            print("---- Initial FRP generated!")
        elif code == 1:
            print("No available or unknown initial FRP. Model Terminated!")
            exit()


# ---- Running the Model ----
for i in np.arange(TT):
    print("--------------------")
    print("---- Cycle t+" + str(i + 1), time[i].strftime("%Y%m%d%H%M"), " running...")

    f_input = (
        "./input/"
        + time_start
        + "/"
        + model_input
        + "."
        + time_start
        + ".f"
        + ("%02i" % i)
        + ".nc"
    )
    f_output = (
        "./output/"
        + time_start
        + "/"
        + model_output
        + "."
        + time_start
        + ".f"
        + ("%02i" % (i + 1))
        + ".nc"
    )

    print("Initial FRP:", f_frp)
    print("Model input:", f_input)
    print("Model output:", f_output)

    # generate model input based on gridded frp
    if opt_inputgen:
        import fire_inputgen

        code = fire_inputgen.main_driver(
            initial_time, i, f_frp, f_input, lat_lim, lon_lim
        )
        if code == 0:
            print("---- Input generated!")
        elif code == 1:
            print("---- No available input. Model terminated!")
            exit()
        elif code == 2:
            print("---- No available fire frame. Model terminated!")
            exit()

    # run spread forecast + post-process
    if opt_forecast:
        import fire_forecast

        fire_forecast.spread_forecast(f_input, f_output)
        print("---- Spread prediction generated!")

    # intensity forecast
    if opt_intensity:
        import fire_intensity

        fire_intensity.main_driver(f_input, f_output, i + 1)
        print("---- Intensity prediction generated!")

    # generate predicted fire maps
    if opt_mapgen:
        from fire_mapgen import fire_mapper

        fire_mapper(f_frp, f_output, opt_intensity)
        print("---- Fire map generated!")

    # simple model evaluation
    if opt_evaluation:
        import fire_evaluation

        code = fire_evaluation.main_driver(
            f_output, frp_source, opt_intensity, lat_lim, lon_lim
        )
        if code == 0:
            print("---- Model evaluation saved!")
        elif code == 1:
            print("---- No available or unknown initial FRP. Model terminated!")
            exit()
        elif code == 2:
            print("---- No matched fires. Skip evaluation!")

    # prepare for next cycle
    if i != TT - 1:
        f_frp = np.copy(f_output)

    print("---- Cycle t+" + str(i + 1), time[i].strftime("%Y%m%d%H%M"), " complete!!!")
    del [f_input, f_output]


# ---- Log saving ----
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


f_log = (
    "./output/"
    + time_start
    + "/"
    + model_output
    + "."
    + time_start
    + "_"
    + time_end
    + ".log"
)

sys.stdout = Logger(f_log)
