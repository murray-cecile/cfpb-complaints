'''
CREATE DATASETS FOR DEEP LEARNING MODELS
'''

import pandas as pd 


SEED = 2020
RAW_FILE = "data/complaints.csv"

NARRATIVE_COL = 'Consumer complaint narrative'
RESPONSE_COL = 'Company response to consumer'
START_DATE = '2016-01-01'
END_DATE = '2020-01-01'


def clean_raw_file(df, ):
    '''
    Keep only rows inside date window with consumer narratives

    Takes: raw complaints df
    Returns: cleaned complaints df
    '''
    
    # Drop missing values in critical columns 
    df.dropna(subset=[NARRATIVE_COL, RESPONSE_COL], inplace=True)

    # Limit to complaints in specified date range 
    df = df[(df['Date received'] >= START_DATE) & (df['Date received'] < END_DATE)]

    return df


def get_n_rows(df, N, keep = False):
    '''
    Get N complaints in shuffled order
    Optionally drop all but complaint text, company response, and ID

    Takes: data frame, integer number of rows to keep
    Returns: data frame with N rows and only useful columns
    '''

    if keep:
        return df.sample(frac=1).iloc[:N, :]

    keep_cols = ['Consumer complaint narrative', \
             'Company response to consumer', \
             'Complaint ID']

    return df.filter(keep_cols).sample(frac=1).iloc[:N, :]


if __name__ == "__main__":

    # clean data frame and shuffle the rows
    full_df = clean_raw_file(pd.read_csv(RAW_FILE))

    # make dev datasets
    # get_n_rows(full_df, 500, keep=True).to_csv("data/complaints_500.csv", index=False)
    # get_n_rows(full_df, 1000, keep=True).to_csv("data/complaints_1k.csv", index=False)
    # get_n_rows(full_df, 3000, keep=True).to_csv("data/complaints_3k.csv", index=False)
    # get_n_rows(full_df, 5000, keep=True).to_csv("data/complaints_5k.csv", index=False)
    # get_n_rows(full_df, 10000, keep=True).to_csv("data/complaints_10k.csv", index=False)

    # prepare to make full train validation test split: shuffle rows
    full_df = full_df.sample(frac=1, random_state=SEED) 
    
    # set training, validation, test size
    train_size = round(0.7 * full_df.shape[0])
    validation_size = round(0.2 * full_df.shape[0])
    test_size = round(0.1 * full_df.shape[0])

    # now make splits
    train_df = full_df[:train_size]
    validation_df = full_df[train_size:train_size + validation_size]
    test_df = full_df[train_size + validation_size:]

    # write to csv
    train_df.to_csv("data/full_training_set.csv", index=False)
    validation_df.to_csv("data/full_validation_set.csv", index=False)
    test_df.to_csv("data/full_testing_set.csv", index=False)




