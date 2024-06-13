from scipy.interpolate import interp1d

class DataInterpolator:
    def __init__(self, df, x_col, y_col):
        self.df = df
        self.x_col = x_col
        self.y_col = y_col

    def interpolate(self, kind='linear'):
        f = interp1d(self.df[self.x_col], self.df[self.y_col], kind=kind)
        return f
