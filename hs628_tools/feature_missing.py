

import numpy as np
class FeatureRemover():

    def __init__(self):

        # Numpy arrays recording information about features to remove
        self.missing_col = None
        self.single_unique = None
        self.collinear_col = None
        self.low_importance = None



    def which_missing(self, data, missing_thresh=0.4):

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




    def colinear(self, data, corr_thresh = 0.9):
        corr_matrix = np.zeros((data.shape[1], data.shape[1]))
        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                corr_matrix[i, j] = np.corrcoef(data[:,i],data[:,j])[0,1]
        corr_pairs = []
        for i in range(corr_matrix.shape[0]):
            for j in range(i+1, corr_matrix.shape[0]):
                if abs(corr_matrix [i,j]) > corr_thresh:
                    corr_pairs += [(i,j)]
        print ('%d features with a correlation greater than %0.2f:\n\n' %(len(corr_pairs), corr_thresh))
        for i in corr_pairs:
            print('columns #%d and #%d have a correlation value of %s \n' %(i[0],i[1], corr_matrix[i[0],i[1]]))
        self.collinear_col = corr_pairs
        return self.collinear_col
