

class Settings:

    def __init__(self):
        self.remove_covariate_shift = False
        self.remove_outliers = True
        self.standardize = True
        self.drop_nan = True
        self.qthr = 0  # threshold for data quality
        self.kfoldCV = 3   # k-fold cross-validation
        self.repeatCV = 30  # repeat inner loop CV
        self.kfoldDLCV = 3  # k-fold cross-validation outer loop
        self.repeatDLCV = 30  # repeat cross validation
        self.prob_calib_alg = 'rank'  # probability calibration
        # self.prob_calib_alg = 'median_centered'  # probability calibration
        # self.prob_calib_alg = 'none'  # probability calibration

    def print_(self):
        print '#### SETTINGS ####'
        print '  remove covariate shift: ', self.remove_covariate_shift
        print '  remove outliers:        ', self.remove_outliers
        print '  standardize:            ', self.standardize
        print '  drop nan:               ', self.drop_nan
        print '  data quality thr:       ', self.qthr
        print '  SLCV k-fold CV:         ', self.kfoldCV
        print '  SLCV repeat k-fold CV:  ', self.repeatCV
        print '  DLCV k-fold CV:         ', self.kfoldDLCV
        print '  DLCV repeat k-fold CV:  ', self.repeatDLCV
        print '  Probability calibration ', self.prob_calib_alg
        print '----------------------------------'
        return ''

