import numpy as np
from numpy.core.multiarray import ndarray
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from typing import List, Any

"""

There are four functions in this package:

1. which_missing: Returns columns with a missing percentage greater than a specified threshold
2. single_value: Returns columns with a single unique value
3. collinear: Returns collinear columns with a correlation greater than a specified correlation threshold
4. low_importance: Returns features that are ranked low importance in the Random Forest Regressor or Classifier

#Notes:  Adapted from:
        https://github.com/WillKoehrsen/feature-selector/blob/master/Feature%20Selector%20Development.ipynb but
        coding with numpy instead of Pandas and using different techniques.
"""


def which_missing(data, missing_thresh=0.4):
    # type: (ndarray, float) -> ndarray

    """
    Find and print information columns with a fraction of missing data more than pre-defined threshold.
    Then plot all the columns with their missing value ratio against the threshold line.
        -----------
        Parameters:
            data: numpy ndarray, shape = [n_samples, n_features]
                    observations and features
            missing_thresh: float between 0 and 1
                    Pre-defined threshold set by the user, default value = 0.4
        -----------
        Returns:
            An array containing the indices of columns with missing values more than the threshold
    """

    # Calculating the fraction of missing values in each column
    missing_series = (len(data) - np.sum(data == data, axis=0)) / float(len(data))  # type: ndarray

    # Find the columns with a missing fraction above the threshold
    missing_col = np.where(missing_series > missing_thresh)
    # List of indices is located in the 0 element of missing_col array
    missing_col = missing_col[0]
    x = np.array([x for x in range(0, data.shape[1])])
    plt.bar(x, missing_series)
    plt.ylabel('missing value ratio')
    plt.xlabel('Feature indices with missing values')
    plt.xticks(x)
    plt.axhline(missing_thresh, color='red', linewidth=5)
    plt.show()
    print('%d columns with more than %0.2f missing values:\n\n' % (len(missing_col), missing_thresh))
    return missing_col


def single_value(data):
    # type: (ndarray) -> ndarray

    """
    Find and print columns that have only a single value.
        -----------
        Parameters:
            data: numpy ndarray, shape = [n_samples, n_features]
                    observations and features
        -----------
        Returns:
            A list containing the indices of columns with only a single value
    """

    single_unique = []
    for i in range(data.shape[1]):
        # finding the unique value of each column and if the length of that equals 1 means
        # that feature has only one single value, then save the index of that column in single_unique
        if len(np.unique(data[:, i])) == 1:
            single_unique += [i]

    single_unique = np.array(single_unique)
    print('%d columns with a single unique value:\n\n' % len(single_unique))
    for i in range(len(single_unique)):
        print('Column # %d with a single value of %s \n' % (single_unique[i], data[1, single_unique[i]]))

    return np.array(single_unique)


def collinear(train_data, corr_thresh=0.9):
    # type: (ndarray, float) -> ndarray

    """
    Find and print columns with correlation coefficient value more than specified value
        -----------
        Parameters:
            train_data: numpy ndarray, shape = [n_samples, n_features]
                    observations and features
            corr_thresh: float between 0 and 1
                    Pre-defined threshold set by the user, default value = 0.9
        -----------
        Returns:
            A list of tuples containing a pair of column indices with
            correlation coefficient value more than specified value
    """

    # creating a matrix for correlation coefficient
    corr_matrix = np.corrcoef(train_data, rowvar=False)

    # finding pairs with corrcoef more than corr_thresh and save them as a tuple in corr_pairs
    corr_pairs = []
    for i in range(corr_matrix.shape[0]):
        for j in range(i + 1, corr_matrix.shape[0]):
            if abs(corr_matrix[i, j]) > corr_thresh:
                corr_pairs += [(i, j)]

    print('%d feature pairs with a correlation greater than %0.2f:\n\n' % (len(corr_pairs), corr_thresh))
    for i in corr_pairs:
        print('columns #%d and #%d have a correlation value of %s \n' % (i[0], i[1], corr_matrix[i[0], i[1]]))
    collinear_col = np.array(corr_pairs)
    return collinear_col


def low_importance(features, target, target_type="classification", importance_thresh=None, n_folds=None):
    # type: (ndarray, ndarray, str, float) -> ndarray

    """
    Find and print columns with their normalized importance value below pre-defined threshold using
    RandomForestRegressor or RandomForestClassifier based on the target_type.
        -----------
        Parameters:
            features :  numpy ndarrays, , shape = [n_samples, n_features]
                        Train data, observations and features

            target:     numpy ndarrays, shape = (1, ) target can be binary if target_type is classification or
                        continuous if target_type is regression

            target_type : string defines the RandomForest method = "regression" if target is continuous or
                        "classification" if target is binary.
                        default value = "classification"

            importance_thresh : float between 0 and 1
                        Pre-defined threshold set by the user, default None: value = 1/number of the features

            n_folds:
        -----------
        Returns:
            A ranked (sorted) list containing indices of features  with importance value below importance_thresh
            :rtype: ndarray


    """
    # Selecting the model based on target_type

    if importance_thresh is None:
        importance_thresh = 1 / features.shape[1]

    if n_folds is None:
        if (features.shape[0] < 100) and (features.shape[0] > 20):
            n_folds = 5
        elif features.shape[0] < 21 and (features.shape[0] > 3):
            n_folds = 3
        elif features.shape[0] < 4:
            n_folds = 1
        else:
            n_folds = 10

    if target_type.lower() == "regression":
        model = RandomForestRegressor()
    elif target_type.lower() == "classification":
        model = RandomForestClassifier()
    else:
        raise ValueError('Target type must be "regression" or "classification" ')

    feature_importance = [0] * features.shape[1]
    # creating k-fold cross validation with 10 folds
    skf = StratifiedKFold(n_splits=n_folds, random_state=None, shuffle=False)
    for train_index, test_index in skf.split(features, target):
        features_train = features[train_index]
        targets_train = target[train_index]
        features_test = features[test_index]
        targets_test = target[test_index]
        # Fitting the model
        model.fit(features_train, targets_train)
        model.score(features_test, targets_test)
        feature_importance += model.feature_importances_
    # normalizing the feature importance values
    feat_imp_norm = feature_importance / skf.get_n_splits(features, target)
    # Creating a template (copy) to sort low important features
    feat_imp_temp = feat_imp_norm.copy()
    low_imp = []

    for i in range(feat_imp_temp.shape[0]):
        # finding the index of the minimum value and save it to low_importance
        low_imp += [feat_imp_temp.argmin()]
        # set the minimum to a large number in feat_imp_norm to find the next minimum value
        feat_imp_temp[feat_imp_temp.argmin()] = 1000

    low_importance_col = []
    for i in range(len(low_imp)):
        if feat_imp_norm[low_imp[i]] < importance_thresh:
            print('Column #%d is #%d in the list of low importance features with the importnace value of %s \n' % (
                low_imp[i], i + 1, feat_imp_norm[low_imp[i]]))
            low_importance_col += [low_imp[i]]

    low_importance_col = np.array(low_importance_col)
    return low_importance_col
