#==============================================================================#
# CFPB COMPLAINTS: EXTRACT/TRANSFORM/LOAD DATA
#
# Nora Hajjar, Cecile Murray, Erika Tyagi
#==============================================================================#

import spacy
import torch
import numpy as np
import pandas as pd 

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
@ignore_warnings(category=FutureWarning)

COMPLAINTS_CSV = "data/complaints.csv"


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

# TO DO: LOAD INTO TORCH FORMAT

def tokenize():
    '''
    '''    
    pass


if __name__ == "__main__":
    
    pass

