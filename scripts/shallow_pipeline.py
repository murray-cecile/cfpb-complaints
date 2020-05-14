import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, make_scorer, classification_report 
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE, RandomOverSampler


# COMPLAINTS_CSV = 'http://files.consumerfinance.gov/ccdb/complaints.csv.zip'
COMPLAINTS_CSV = '../data/complaints.csv'


def load_data(verbose=False):
    '''
    Loads and pre-processes data from the CFPB complaints database 
    Returns: pandas df
    '''
    # Load local csv - also accessible from http://files.consumerfinance.gov/ccdb/complaints.csv.zip 
    df = pd.read_csv(COMPLAINTS_CSV)
    
    # Drop missing values in critical columns 
    df.dropna(subset=['Consumer complaint narrative', 
                      'Company response to consumer'], inplace=True)
    
    # Clean column names 
    new_names = [n.lower().replace(" ", "_").replace("?", "") for n in df.columns]
    df.columns = new_names

    # Limit to complaints between 2016 and 2020 
    df = df[(df.date_received >= '2016-01-01') & (df.date_received < '2020-01-01')]
    
    # Convert dates 
    df.date_received = pd.to_datetime(df.date_received)
    df.date_sent_to_company = pd.to_datetime(df.date_sent_to_company)

    # Create week variable 
    df['week'] = df['date_received'].dt.strftime('%Y-w%V')
        
    # Print summary information 
    if verbose: 
        print('Date range: {} to {}'.format(
            df['date_received'].min(), df['date_received'].max()))
        print('Number of complaints: {}'.format(
            df.shape[0]))
        print('\nDistribution of company response: \n{}'.format(
            df['company_response_to_consumer'].value_counts() / df.shape[0]))
        print('\nNumber of unique values in each column: \n{}'.format(
            df.nunique()))
        
    return df


def process_features(df, cat_columns, narrative_col): 
    '''
    '''
    # Process categorical features 
    return_df = process_cat_features(df, cat_columns)

    # Process narrative - keep indices aligned 
    char_count, word_count = process_narrative(df, narrative_col)
    return_df = return_df.assign(char_count=char_count.values)
    return_df = return_df.assign(word_count=word_count.values)

    return return_df


def process_cat_features(df, cat_columns):
    '''
    Processes categorical features
    Input: 
    - df: pandas df 
    - cat_columns: iterable with tuples where the first value is the name of a 
        column and the second value is the maximum number of values to one-hot-encode 
        e.g. [('product', 50), ('sub-product', 50)] 
    Returns: pandas df 
    '''
    # Intialize output features dataset 
    return_df = pd.DataFrame() 
    
    for col_name, max_cats in cat_columns: 

        # Replace values with Missing (if originally NA) or Other (if < max_cats)
        if not max_cats: 
            max_cats = df[col_name].nunique()

        top_n = df[col_name].value_counts()[:max_cats].index.tolist()
        return_df[col_name] = np.where(df[col_name].isin(top_n), df[col_name], 
                                       np.where(df[col_name].isna(), 'Missing', 'Other')) 
        
        # One-hot encode features 
        return_df = pd.get_dummies(return_df, columns=[col_name])
        
    return return_df


def process_narrative(df, narrative_col): 
    '''
    Processes the narrative text field 
    Input: 
    - df: pandas df
    - narrative_col: name of the narrative column to process 
    Returns: 
    - character count (pandas series), word count (pandas series) 
    '''
    char_count = df[narrative_col].str.len()
    word_count = df[narrative_col].apply(lambda x: len(x.split()))
    
    return char_count, word_count


def split_resample(X, y, resample=False, verbose=False): 
    '''
    Train-test-splits and resamples using SMOTE 
    Input: 
    - X: pandas df or numpy array (features)
    - y: pandas series or numpy array (label)
    - resample: boolean indicating whether to perform resampling 
    Returns: 
    - X_train, X_test, y_train, y_test
    '''
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0)

    # Resample training data to balance classes 
    if resample: 
        sm = SMOTE(random_state=0) 
        X_train, y_train = sm.fit_sample(X_train, y_train)

    if verbose: 
        print('\nDistribution of training labels: \n{}'.format(
            y_train.value_counts() / y_train.shape[0]))

    return X_train, X_test, y_train, y_test


def hypertune_RF(X_train, y_train, params, scorer_metric=balanced_accuracy_score, verbose=False): 
    '''
    Performs a grid serach to optimize the hyperparameters over params 
    Input: 
    - X_train, y_train 
    - params: dictionary of parameters and values 
        e.g. {"estimator__n_estimators": [10], "estimator__max_depth": [5]} 
    - scorer_metric: metric from sklearn.metrics
    '''
    scorer = make_scorer(scorer_metric)
    rf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
    grid = GridSearchCV(rf, param_grid=params, scoring=scorer)

    grid.fit(X_train, y_train)

    if verbose: 
        print('\nBest score:', grid.best_score_)
        print('Best parameters:', grid.best_params_)

    return grid.best_estimator_


def summarize_probs(clf, y_test, y_pred):
    '''
    Summarizes predictions (i.e. highest probability class)
    '''
    summary = classification_report(y_test, y_pred, zero_division=0)
    print(summary)  


def summarize_probas(clf, y_test, y_pred_proba): 
    '''
    Summarizes prediction probabilities 
    '''
    proba_matrix = pd.DataFrame(y_pred_proba)
    proba_matrix.columns = clf.classes_
    proba_matrix = proba_matrix.assign(y_test=y_test.values)
    summary = proba_matrix.groupby('y_test').agg(lambda x: np.mean(x) * 100).round(3)

    return summary 


def feature_importance(clf, X_train, verbose=False): 
    '''
    Returns dictionary summarizing feature importance for each class 
    '''
    feat_imp_dict = {} 

    for i, label in enumerate(clf.classes_):
        coefs = pd.DataFrame({'feature': X_train.columns.values, 
                              'importance': clf.estimators_[i].feature_importances_.ravel()})
        coefs = coefs.sort_values(by='importance', ascending=False)
        
        if verbose: 
            print('\nClass:', label)
            print(coefs[:5])
        
        feat_imp_dict[label] = coefs
        
    return feat_imp_dict




