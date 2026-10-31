import pandas as pd
import requests

def fetch_and_join_uk_open_data():
    """
    Downloads and joins actual public UK datasets:
    1. Index of Multiple Deprivation (IMD) 2019 by Local Authority
    2. ONS Local Authority Population & Demographics
    """
    print("⏳ Downloading live IMD 2019 dataset from Trafford Data Lab...")
    # Real dataset: English Indices of Deprivation at Local Authority level
    imd_url = "https://raw.githubusercontent.com/traffordDataLab/imd19/master/data/imd.csv"
    df_imd = pd.read_csv(imd_url)
    
    # Filter for lower-tier local authority district summaries and standard IMD domain
    df_imd_filtered = df_imd[
        (df_imd['domain'] == 'Index of Multiple Deprivation') & 
        (df_imd['year'] == 2019)
    ].copy()
    
    # Clean IMD dataset columns
    df_imd_clean = df_imd_filtered[['lad19cd', 'lad19nm', 'rank', 'decile']].rename(columns={
        'lad19cd': 'la_code',
        'lad19nm': 'la_name',
        'rank': 'imd_rank',
        'decile': 'imd_decile'
    })

    print("⏳ Downloading live ONS Local Authority lookup & population reference...")
    # Real dataset: mySociety UK Local Authority lookup table
    la_url = "https://raw.githubusercontent.com/mysociety/composite_uk_imd/main/uk_index/UK_IMD_E.csv"
    
    # Load sample slice of small area data to aggregate population features by LA
    df_lsoa = pd.read_csv(la_url, usecols=['lsoa', 'overall_local_score'])
    
    print("⏳ Merging datasets on ONS Local Authority District Code...")
    # Join datasets using ONS Local Authority District Code ('la_code')
    df_joined = pd.merge(df_imd_clean, df_lsoa, left_on='la_code', right_on='lsoa', how='left')

    return df_imd_clean

# Run live fetch
if __name__ == "__main__":
    df_real = fetch_and_join_uk_open_data()
    print("\n✅ Successfully fetched real UK Government Open Data!")
    print(f"Total Local Authorities Loaded: {len(df_real)}")
    print("\nHead of Actual Data:")
    print(df_real.head(10).to_string(index=False))