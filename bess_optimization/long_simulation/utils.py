import ExcelOpener_l
import Interpolator_l

class Get_data:
    @staticmethod
    def get_data(file_path2, sheetname3):
        # Import data from an Excel file and a specific sheet using a method from the ExcelOpener module.
        data = ExcelOpener_l.import_file.load_excel(file_path2, sheetname3)
        # Return the imported data.
        return data

class BESS:

    @staticmethod
    def get_bess(technology, properties, se_sp, size):
        # Select specific columns from the properties DataFrame.
        BESS_Parameters = properties.iloc[:, [0, 1, 7, 9, 14, 15]]
        # Filter the DataFrame for the specified technology.
        BESS_Parameters = BESS_Parameters[BESS_Parameters['Technology'] == technology]
        # Further filter the DataFrame for the specified specific energy/specific power.
        BESS_Parameters = BESS_Parameters[BESS_Parameters['Specific Energy / Specific Power'] == se_sp]
        # Add a new column for the size in kWh.
        BESS_Parameters['Size [kWh]'] = size
        # Select specific columns to keep in the final DataFrame.
        BESS_Parameters = BESS_Parameters.iloc[:, [0, 1, 2, 3, 6]]

        # Return the filtered and modified DataFrame.
        return BESS_Parameters

    @staticmethod
    def get_c_d_functions(load_curve):
        # Select charge rate DataFrame from the first 356 rows of load_curve.
        charge_rate = load_curve.iloc[:356, [0, 3.5]]
        # Select discharge rate DataFrame from rows 357 onwards of load_curve.
        discharge_rate = load_curve.iloc[357:, [0, 4, 5]]
        # Interpolate data for the charge rate.
        charge_interpolator = Interpolator_l.DataInterpolator(charge_rate, 'SoC [%]', 'Charge Rate [kWh/(kWhp*h)]')
        charge_rate_interpolated_func = charge_interpolator.interpolate()
        # Interpolate data for the discharge rate.
        discharge_interpolator = Interpolator_l.DataInterpolator(discharge_rate, 'SoC [%]', 'Discharge Rate [kWh/(kWhp*h)]')
        discharge_rate_interpolated_func = discharge_interpolator.interpolate()

        # Return the interpolated functions for charge and discharge rates.
        return charge_rate_interpolated_func, discharge_rate_interpolated_func



def get_charged_energy(c_d, soc, size, soc_max, c_func):
    """
    Calculate the amount of energy that can be charged into the BESS at a given timestep.

    Parameters:
    - c_d (float): Charge/discharge signal at current timestep.
    - soc (float): Current state of charge (SOC) of the battery.
    - size (float): Battery size or power capacity.
    - soc_max (float): Maximum allowed SOC (typically 1.0 or 100%).
    - c_func (function): Charging power limitation function depending on SOC.

    Returns:
    - float: Amount of energy charged into the BESS.
    """

    if c_d > 0:
        energy = min(c_d * size, c_func(soc) * size)
        energy = min(energy, max((soc_max - soc) * size, 0.0))
        return energy
    else:
        return 0.0

def get_discharged_energy(c_d, soc, size, d_func, soc_min):
    """
    Computes the discharged energy from the BESS, respecting power and SOC constraints.

    Parameters:
    - c_d: desired discharge power value (float)
    - soc: current state of charge (float)
    - size: system size (float)
    - d_func: discharge limit function that returns a negative value based on SOC
    - soc_min: minimum allowed SOC (float)

    Returns:
    - discharged energy (float)
    """
    max_discharge_power = max(c_d * size, -d_func(soc) * size)
    soc_limit = min((soc_min - soc) * size, 0.0)

    if c_d < 0.0:

        energy = max(max_discharge_power, soc_limit)

    else:

        energy = 0.0

    return energy
