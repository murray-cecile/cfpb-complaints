import warnings
import pandas as pd 
import numpy as np

from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Input files 
# Also accessible from http://files.consumerfinance.gov/ccdb/complaints.csv.zip 
COMPLAINTS_CSV = 'http://files.consumerfinance.gov/ccdb/complaints.csv.zip'

# Output files 
FEATURE_IMPORTANCE_OUT = 'results/feature_importance.json'
MODEL_EVALUATION_OUT = 'results/model_evaluation.csv'
MODEL_EVALUATION_OUT_FULL = 'results/model_evaluation_full.csv'
PROCESSED_DATA_DIR = 'processed_data'

# Column names 
RESPONSE_COL = 'company_response_to_consumer'
NARRATIVE_COL = 'consumer_complaint_narrative'
DATE_COL = 'date_received'

# Date range for complaints 
START_DATE = '2016-01-01'
END_DATE = '2020-01-01'

# Categorical features, maximum number of values to one-hot encode 
CAT_COLUMNS = [
    ('product', 50), 
    ('sub-product', 50), 
    ('issue', 50), 
    ('sub-issue', 50), 
    ('company', 100), 
    ('state', None), 
    ('tags', None), 
    ('week', None)
]

# Continuous features  
CONTINUOUS_COLUMNS = [
    'char_count', 
    'word_count'
]

# Fraction of the majority class to sample 
FRAC_MAJORITY = 0.2 

# k in k-fold cross validation 
K = 5

# Cross validation evaluation metric 
CV_METRIC = f1_score
CV_AVERAGE = 'micro'

# Models 
MODELS = {
    'LR': OneVsRestClassifier(LogisticRegression(max_iter=1000)), 
    'RF':  OneVsRestClassifier(RandomForestClassifier()), 
    'SVM': OneVsRestClassifier(SVC(probability=True))
}

PARAMETERS = {
    'LR':  {'estimator__penalty': ['l2'], 
            'estimator__C': [0.1, 1.0, 10]}, 
    'RF':  {'estimator__n_estimators': [int(x) for x in np.linspace(start=10, stop=200, num=5)], 
           'estimator__max_depth': [int(x) for x in np.linspace(start=10, stop=200, num=5)]}, 
    'SVM': {'estimator__C': [0.1, 1.0, 10], 
            'estimator__kernel': ['linear']} 
}


