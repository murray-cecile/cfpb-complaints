'''
Create small datasets for dev 
'''

import pandas as pd 

RAW_FILE = "data/complaints.csv"

# to do: implement a random sampling function

if __name__ == "__main__":

    full_df = pd.read_csv(RAW_FILE)

    #drop rows where consumer complaints missing
    full_df.dropna(subset=['Consumer complaint narrative'], inplace=True)

    full_df.iloc[1:1000,:].to_csv("data/complaints_1k.csv", index=False)
    full_df.iloc[1000:1500,:].to_csv("data/complaints_500.csv", index=False)


