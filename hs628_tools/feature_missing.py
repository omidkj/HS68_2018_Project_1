

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

        # Calculating the fraction of missing values in each column
        missing_series = (len(a) - np.sum(data==data, axis=0)) / float(len(data))

        # Find the columns with a missing fraction above the threshold
        missing_col = np.where(missing_series > missing_thresh)

        missing_col = missing_col[0]

        self.missing_col = missing_col

        print('%d columns with more than %0.2f missing values:\n\n' % (len(self.missing_col), self.missing_thresh))
        for i in range(len(missing_col)):
                print ('Colunm # %d with %0.2f missing values.\n' % (missing_col[i], missing_series[missing_col[i]]))

        return missing_col

    def single_value(self, data):
        single_unique = []
        for i in range(data.shape[1]):
            if len(np.unique(data[:,i], axis=0)) == 1:
                single_unique += [i]

        self.single_unique = np.array(single_unique)
        print('%d columns with a single unique value:\n\n' % len(self.single_unique))
        for i in range(len(self.single_unique)):
            print('Colunm # %d with a single value of %s \n' % (single_unique[i], data[1, single_unique[i]]))

        return np.array(single_unique)
   
