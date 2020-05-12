import numpy as np
import pandas as pd 

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)


# COMPLAINTS_CSV = 'http://files.consumerfinance.gov/ccdb/complaints.csv.zip'
COMPLAINTS_CSV = '../data/complaints.csv'
COMPLAINT_COL = 'consumer_complaint_narrative'


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
    
    # Convert dates 
    df.date_received = pd.to_datetime(df.date_received)
    df.date_sent_to_company = pd.to_datetime(df.date_sent_to_company)
    
    # Limit to complaints before 2020 
    df = df[df.date_received < '2020-01-01']
    
    # Print summary information 
    if verbose: 
        print('Date range: {} to {}'.format(
            df['date_received'].min(), df['date_received'].max()))
        print('Number of complaints: {}'.format(
            df.shape[0]))
        print('\nDistribution of company response: \n{}'.format(
            df['company_response_to_consumer'].value_counts() / df.shape[0]))
        print('\nDistribution of missing values: \n{}'.format(
            df.isna().sum() / df.shape[0]))
        print('\nNumber of unique values in each column: \n{}'.format(
            df.nunique()))
        
    return df


def process_cat_features(df, columns):
    '''
    Processes categorical features
    Input: 
    - df: pandas df 
    - columns: iterable object where the first value is the name of a column 
        and the second value is the maximum number of values to one-hot-encode 
        e.g. {('product', 50), ('sub-product', 50)} 
    Returns: pandas df 
    '''

    # Intialize output features dataset 
    return_df = pd.DataFrame() 
    
    for c in columns: 
        col_name = c[0]
        if c[1]: 
            max_cats = c[1] 
        else: 
            max_cats = df[col_name].nunique()
        
        # Replace values with Missing (if originally NA) or Other (if < than top_n)
        top_n = df[col_name].value_counts()[:max_cats].index.tolist()
        return_df[col_name] = np.where(df[col_name].isin(top_n), df[col_name], 
                                       np.where(df[col_name].isna(), 'Missing', 'Other')) 
        
        # One-hot encode features 
        return_df = pd.get_dummies(return_df, columns = [col_name])
    
    return return_df





