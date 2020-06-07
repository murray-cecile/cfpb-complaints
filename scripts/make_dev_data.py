'''
Create small datasets for dev 
'''

import pandas as pd 

RAW_FILE = "data/complaints.csv"


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

    full_df = pd.read_csv(RAW_FILE)

    #drop rows where consumer complaints missing
    full_df.dropna(subset=['Consumer complaint narrative'], inplace=True)

    # get_n_rows(full_df, 1000, keep=True).to_csv("data/complaints_1k.csv", index=False)
    # get_n_rows(full_df, 500).to_csv("data/complaints_500.csv", index=False)

    # get_n_rows(full_df, 10000, keep=True).to_csv("data/complaints_10k.csv", index=False)
    # get_n_rows(full_df, 5000, keep=True).to_csv("data/complaints_5k.csv", index=False)

    # get_n_rows(full_df, 3000, keep=True).to_csv("data/complaints_3k.csv", index=False)
    get_n_rows(full_df, 500, keep=True).to_csv("data/complaints_500.csv", index=False)

