#==============================================================================#
# CFPB COMPLAINTS: EXPLORATORY DATA ANALYSIS
#
#==============================================================================#

# import spacy
import numpy as np
import pandas as pd 

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

COMPLAINTS_CSV = "data/complaints.csv"
COMPLAINT_COL = 'consumer_complaint_narrative'


def load_data():
    '''
    Loads csv file, drops complaints without consumer narrative, 
     renames columns
    Takes: none
    Returns: pandas df
    '''

    df = pd.read_csv(COMPLAINTS_CSV)
    df.dropna(subset=['Consumer complaint narrative'], inplace=True)
    
    new_names = [n.lower().replace(" ", "_").replace("?", "") for n in df.columns]
    df.columns = new_names

    return df


# def tokenize_narratives(df):
#     '''
#     DO NOT USE ON FULL DF - chunks better
#     Takes: complaint dataframe
#     Returns: df with column of tokenized narrative
#     '''

#     df['tokenized'] = df['consumer_complaint_narrative'].apply(lambda x: nlp(x))
#     return df


def summarize_narrative_length(df):
    '''
    Computes complaint length in characters and # of spaces
    Prints the summary stats
    Takes: complaint dataframe
    Returns: df with summary cols
    '''

    subdf = df[['complaint_id', COMPLAINT_COL]]
    subdf['complaint_char_len'] = subdf[COMPLAINT_COL].str.len()
    subdf['complaint_space_num'] = subdf[COMPLAINT_COL].str.count(" ")

    cols = ['complaint_char_len', 'complaint_space_num']    

    print(subdf[cols].describe())

    return subdf

if __name__ == "__main__":
    
    df = load_data()

    stats = summarize_narrative_length(df)
    stats.to_csv("data/complaint_stats.csv")

    # NEED TO DOWNLOAD THIS en_core_web_sm thing, 
    # python3 -m spacy download en_core_web_sm
    # nlp = spacy.load("en_core_web_sm")
