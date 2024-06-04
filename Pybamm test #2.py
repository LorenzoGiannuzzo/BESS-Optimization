import pandas as pd
import numpy as np
import ExcelOpener
import pybamm
import matplotlib

matplotlib.use('TkAgg')

model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model)
sim.solve([0, 3600])

variables = model.variable_names()
output_variables = ["C-rate","Discharge energy [W.h]","Throughput energy [W.h]"]
sim.plot(output_variables=output_variables)