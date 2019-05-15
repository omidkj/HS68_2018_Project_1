# HS68_2018_Project_1

This Repository is to facilitate software development & collaboration by USF class HS 628.

There are four functions in this package:
1. which_missing: Returns columns with a missing percentage greater than a specified threshold
2. single_value: Returns columns with a single unique value
3. collinear: Returns collinear columns with a correlation greater than a specified correlation threshold
4. low_importance: Returns features that are ranked low importance in the Random Forest Regressor or Classifier
#Notes:  Adapted from:
        https://github.com/WillKoehrsen/feature-selector/blob/master/Feature%20Selector%20Development.ipynb but
        coding with numpy instead of Pandas and using different techniques.
"""
