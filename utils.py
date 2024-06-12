import ExcelOpener
import Interpolator

class Get_data:
    @staticmethod
    def get_data(file_path2, sheetname3):
        data = ExcelOpener.import_file.load_excel(file_path2, sheetname3)
        return data

class BESS:

    @staticmethod
    def get_bess(technology, properties, se_sp, size):
        BESS_Parameters = properties.iloc[:, [0, 1, 7, 9, 14, 15]]
        BESS_Parameters = BESS_Parameters[BESS_Parameters['Technology'] == technology]
        BESS_Parameters = BESS_Parameters[BESS_Parameters['Specific Energy / Specific Power'] == se_sp]
        BESS_Parameters['Size [kWh]'] = size
        BESS_Parameters = BESS_Parameters.iloc[:, [0, 1, 2, 3, 6]]

        return BESS_Parameters

    @staticmethod
    def get_c_d_functions(load_curve):
        # Select charge_rate and discharge_rate DataFrames from load_curve
        charge_rate = load_curve.iloc[:356, [0, 3.5]]
        discharge_rate = load_curve.iloc[357:, [0, 4, 5]]
        # Interpolate data for charge_rate and discharge_rate
        charge_interpolator = Interpolator.DataInterpolator(charge_rate, 'SoC [%]', 'Charge Rate [kWh/(kWhp*h)]')
        charge_rate_interpolated_func = charge_interpolator.interpolate()
        discharge_interpolator = Interpolator.DataInterpolator(discharge_rate, 'SoC [%]',
                                                               'Discharge Rate [kWh/(kWhp*h)]')
        discharge_rate_interpolated_func = discharge_interpolator.interpolate()

        return charge_rate_interpolated_func, discharge_rate_interpolated_func

