"""

BESS Optimization using NSGA-III Algorithm

    __author__ = "Lorenzo Giannuzzo"
    __maintainer__ = "Lorenzo Giannuzzo"
    __email__ = "lorenzo.giannuzzo@polito.it"
    __status__ = "in progress"
    __version__ = "v0.2.1"
    __license__ = "MIT"

Last Update of current code: 09/01/2025 - 17:39

"""

# IMPORT LIBRARIES
from scipy.interpolate import interp1d

# CREATE DATA INTERPOLATOR CLASS

class DataInterpolator:

    def __init__(self, df, x_col, y_col):

        self.df = df
        self.x_col = x_col
        self.y_col = y_col

    def interpolate(self, kind='linear'):

        f = interp1d(self.df[self.x_col], self.df[self.y_col], kind=kind)

        return f
