import pandas as pd
import numpy as np
import ExcelOpener
import pybamm
import matplotlib

matplotlib.use('TkAgg')
sets = list(pybamm.parameter_sets)

# for Li-ion batteries
parameters_values = pybamm.ParameterValues("Ai2020")

experiment = pybamm.Experiment(
    [
        (
            "Discharge at C/10 for 10 hours or until 3.3 V",
            "Rest for 1 hour",
            "Charge at 1 A until 4.1 V",
            "Hold at 4.1 V until 50 mA",
            "Rest for 1 hour",
        )
    ]
    * 3
    + [
        "Discharge at 1C until 3.3 V",
    ]
)

model = pybamm.lithium_ion.DFN()
sim = pybamm.Simulation(model, experiment=experiment, parameter_values = parameters_values)
sim.solve()
#sim.plot()

output_variables = ["C-rate","Power [W]","Terminal power [W]"]
sim.plot(output_variables=output_variables)

