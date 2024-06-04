import pandas as pd
import numpy as np
import ExcelOpener
import pybamm
import matplotlib

#Import a Doyle-Fuller-Newman model
model = pybamm.lithium_ion.DFN()

# Create a simulation

sim = pybamm.Simulation(model)
import pandas as pd
import numpy as np
import ExcelOpener
import pybamm
import matplotlib

# Solve the simulation

models = [
    pybamm.lithium_ion.SPM(),
    pybamm.lithium_ion.SPMe(),
    pybamm.lithium_ion.DFN(),
]

sims =[]
matplotlib.use('TkAgg')
for model in models:
   sim = pybamm.Simulation(model)
   sim.solve([0, 3600])
   sims.append(sim)

pybamm.dynamic_plot(sims)
