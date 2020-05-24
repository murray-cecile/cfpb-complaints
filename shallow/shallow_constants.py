import warnings
import pandas as pd 

from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Input files 
# Also accessible from http://files.consumerfinance.gov/ccdb/complaints.csv.zip 
COMPLAINTS_CSV = '../data/complaints.csv'

# Output files 
FEATURE_IMPORTANCE_OUT = 'results/feature_importance.json'
MODEL_EVALUATION_OUT = 'results/model_evaluation.csv'
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

# k in k-fold cross validation 
K = 5

# Cross validation evaluation metric 
CV_METRIC = f1_score
CV_AVERAGE = 'micro'

# Models 
MODELS = {
    'RF':  OneVsRestClassifier(RandomForestClassifier()), 
    'SVM': OneVsRestClassifier(SVC())
}

# Model parameters 
PARAMETERS = {
    'RF':  {'estimator__n_estimators': [10, 20], 
           'estimator__max_depth': [3, 5, 10]}, 
    'SVM': {'estimator__C': [0.01, 0.1], 
            'estimator__kernel': ['linear', 'rbf']} 
}



