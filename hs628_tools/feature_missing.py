

import numpy as np
class FeatureRemover():

    def __init__(self):

        # Numpy arrays recording information about features to remove
        self.missing_col = None
        self.single_unique = None
        self.collinear = None
        self.low_importance = None

        self.feature_importances = None


    def which_missing(self, data, missing_thresh):

        self.missing_thresh = missing_thresh

        # Calculate the fraction of missing in each column
        missing_series = (len(a) - np.sum(data==data, axis=0)) / float(len(data))

        # Find the columns with a missing percentage above the threshold
        missing_col = np.where(missing_series > missing_thresh)

        missing_col = missing_col[0]

        self.missing_col = missing_col

        print('%d columns with greater than %0.2f missing values.\n' % (len(self.missing_col), self.missing_thresh))
        return missing_col

        

    def single_value(self, data):
        single_unique = []
        for i in range(data.shape[1]):
            if len(np.unique(data[:,i], axis=0)) == 1:
                single_unique += [i]

        self.single_unique = np.array(single_unique)
        print('%d columns with a single unique value.\n' % len(self.single_unique))

        return np.array(single_unique)
