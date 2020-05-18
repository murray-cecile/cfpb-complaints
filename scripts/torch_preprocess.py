'''
PREPROCESS CFPB CONSUMER COMPLAINTS CSV FILE

This script should only need to be run once (per dev dataset). It will:
- load in the full csv archive
- drop rows where the consumer narrative is missing
- drop rows outside desired date range
- convert column names to lowercase with underscores
- convert the date field to a date type
- OPTIONALLY one hot encode categorical variables
- RESAMPLE to correct for class imbalance: CAREFUL, THIS CAN BE SLOW

(It may not make sense to do resample here after all)

Here is what it *doesn't* do:
- drop rows where company response, product type, or issue type are missing
- create any additional variables (week dummy, narrative length, etc)
'''

import argparse
import warnings
import numpy as np
import pandas as pd 
import shallow_pipeline as pipeline

from imblearn.over_sampling import SMOTE

warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

SEED = 2020

START_DATE = '2016-01-01'
END_DATE = '2020-01-01'

RAW_TEXT_COL = 'Consumer complaint narrative'
TEXT_COL = 'consumer_complaint_narrative'
LABEL_COL = 'company_response_to_consumer'
CAT_COLUMNS = [
    ('product', 50), 
    ('sub-product', 50), 
    ('issue', 50), 
    ('sub-issue', 50), 
    ('company', 100), 
    ('state', None), 
    ('tags', None), 
    # ('week', None)
]



def load_drop_convert(path, drop_cat_cols=True):
    '''
    Loads csv file, drops null narrative rows, converts names
    Either drop categorical columns or binarize them

    Takes: string file path
    Returns: feature and label df
    '''

    df = pd.read_csv(path)

    # drop rows where consumer narrative is missing
    df.dropna(subset=[RAW_TEXT_COL], inplace=True)
    
    # rename columns
    new_names = [n.lower().replace(" ", "_").replace("?", "") for n in df.columns]
    df.columns = new_names

    # Limit to complaints to specified dates
    df = df[(df.date_received >= START_DATE) & (df.date_received < END_DATE)]
    
    # Convert dates 
    df.date_received = pd.to_datetime(df.date_received)
    df.date_sent_to_company = pd.to_datetime(df.date_sent_to_company)

    # split into feature and label columns
    features = df.drop(columns=LABEL_COL)
    labels = df[LABEL_COL]

    # either drop or process categorical features
    if drop_cat_cols:
        features.drop(columns=[c for c, i in CAT_COLUMNS], inplace=True)
    else:
        features = pipeline.process_cat_features(features, CAT_COLUMNS)

    return features, labels


def resample(features, labels, verbose=False): 
    '''
    Resamples using SMOTE 
    Input: 
    - features: pandas df or numpy array 
    - labels: pandas series or numpy array 
    - resample: boolean indicating whether to perform resampling 
    Returns: 
    - resampled data frame
    '''

    # make index for complaint id to narrative
    id_to_narrative = features[['complaint_id', TEXT_COL]]
    features_no_text = features.drop(columns=TEXT_COL)

    # Resample training data to balance classes 
    sm = SMOTE(random_state=SEED) 
    resampled, r_labels = sm.fit_sample(features_no_text, labels)

    # now merge text fields back in
    r_features = pd.merge(resampled, id_to_narrative, how="left", on="complaint_id")

    if verbose: 
        print('\nDistribution of training labels: \n{}'.format(
            r_labels.value_counts() / r_labels.shape[0]))

    return pd.concat([r_features.reset_index(drop=True), r_labels], axis=1)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", "--infile", default="data/complaints.csv", help="raw csv file path")
    parser.add_argument("--outfile", "--outfile", help="file path for processed df")
    parser.add_argument("--dropcat", "--dropcat", default=True, help="Boolean: drop categorical columns?")
    args = parser.parse_args()

    keep_cols = [TEXT_COL, LABEL_COL, 'complaint_id']

    features, labels = load_drop_convert(args.infile, args.dropcat)
    features = features.filter(keep_cols) # do not use most fields right now
    # print(features.head())
    resampled  = resample(features, labels, verbose=True)
    resampled.to_csv(args.outfile)

    