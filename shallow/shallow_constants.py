import warnings
import pandas as pd 
import numpy as np

from sklearn.metrics import f1_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB

# Input files 
# COMPLAINTS_CSV = 'http://files.consumerfinance.gov/ccdb/complaints.csv.zip'
COMPLAINTS_CSV = '../data/complaints.csv'

# Date range for complaints 
START_DATE = '2016-01-01'
END_DATE = '2020-01-01'

# Column names 
RESPONSE_COL = 'company_response_to_consumer'
NARRATIVE_COL = 'consumer_complaint_narrative'
DATE_COL = 'date_received'

# k in k-fold cross validation 
K = 5

# ----------  TASK 1 SETUP ---------- # 

# Output files 
FEATURE_IMPORTANCE_OUT = 'results/feature_importance_task1.json'
MODEL_EVALUATION_OUT = 'results/model_evaluation_task1.csv'
MODEL_EVALUATION_OUT_FULL = 'results/model_evaluation_full_task1.csv'
PROCESSED_DATA_DIR = 'processed_data'

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

# Cross validation evaluation metric 
CV_METRIC = f1_score
CV_AVERAGE = 'micro'

# Models and parameter grids 
MODELS = {
    'LR':  OneVsRestClassifier(LogisticRegression(max_iter=1000)), 
    'NB':  OneVsRestClassifier(GaussianNB()), 
    'RF':  OneVsRestClassifier(RandomForestClassifier()), 
    'SVM': OneVsRestClassifier(SVC(probability=True)), 
    'GBT': OneVsRestClassifier(GradientBoostingClassifier())
}

PARAMETERS = {
    'LR':  {'estimator__penalty': ['l2'], 
            'estimator__C': [0.1, 1.0, 10]}, 
    'NB':  {}, 
    'RF':  {'estimator__n_estimators': [int(x) for x in np.linspace(start=10, stop=200, num=5)], 
           'estimator__max_depth': [int(x) for x in np.linspace(start=10, stop=200, num=5)]}, 
    'SVM': {'estimator__C': [0.1, 1.0, 10], 
            'estimator__kernel': ['linear']}, 
    'GBT': {'estimator__learning_rate': [0.1, 1.0, 0.5], 
            'estimator__max_depth': [int(x) for x in np.linspace(start=10, stop=200, num=5)]}
}

# ----------  TASK 2 SETUP ---------- # 

# Naive Bayes parameter grid 
PARAMS_NB = {
    'vect__ngram_range': [(1, 1), (1, 2)], 
    'tfidf__use_idf': (True, False), 
    'clf__alpha': (1e-2, 1e-3)
}

# SVM parameter grid 
PARAMS_SVM = {
    'vect__ngram_range': [(1, 1), (1, 2)], 
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3)
}

# Label column (product or issue)
LABEL_COL = 'product' 





