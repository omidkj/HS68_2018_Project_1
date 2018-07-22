

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier



class FeatureRemover():

    def __init__(self):

        # Numpy arrays recording information about features to remove
        self.missing_col = None
        self.single_unique = None
        self.collinear_col = None
        self.low_importance_col = None



    def which_missing(self, data, missing_thresh=0.4):

        self.missing_thresh = missing_thresh

        # Calculating the fraction of missing values in each column
        missing_series = (len(data) - np.sum(data==data, axis=0)) / float(len(data))

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
        self.corr_thresh = corr_thresh
        corr_matrix = np.zeros((data.shape[1], data.shape[1]))
        for i in range(data.shape[1]):
            for j in range(data.shape[1]):
                corr_matrix[i, j] = np.corrcoef(data[:,i],data[:,j])[0,1]
        corr_pairs = []
        for i in range(corr_matrix.shape[0]):
            for j in range(i+1, corr_matrix.shape[0]):
                if abs(corr_matrix [i,j]) > self.corr_thresh:
                    corr_pairs += [(i,j)]
        print ('%d features with a correlation greater than %0.2f:\n\n' %(len(corr_pairs), self.corr_thresh))
        for i in corr_pairs:
            print('columns #%d and #%d have a correlation value of %s \n' %(i[0],i[1], corr_matrix[i[0],i[1]]))
        self.collinear_col = corr_pairs
        return self.collinear_col


    def low_importance(self, features, target, target_type = "classification", importance_thresh = 1/features.shape[1]):

        self.features = features
        self.target = target
        self.importance_thresh = importance_thresh

        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, test_size = 0.25)

        if target_type == "regression":
            model = RandomForestRegressor()
        elif target_type == "classification":
            model = RandomForestClassifier()
        else:
            raise ValueError('Target type must be "regression" or "classification" ')

        model.fit(X_train, y_train)
        feature_importance = model.feature_importances_
        feat_imp_norm = feature_importance / np.sum(feature_importance)
        feat_imp_temp = feat_imp_norm.copy()
        low_importance = []

        for i in range(feat_imp_temp.shape[0]):
            low_importance += [feat_imp_norm.argmin()]
            feat_imp_norm [feat_imp_norm.argmin()]= 10

        for i in range(len(low_importance)):
            if feature_importance[low_importance[i]] < self.importance_thresh:
                print ('Column #%d is #%d in the list of low importance features with the importnace value of %s \n' %(low_importance[i], i+1, feature_importance[low_importance[i]]))

        self.low_importance_col = low_importance
        return self.low_importance_col
