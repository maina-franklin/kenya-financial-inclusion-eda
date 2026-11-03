import pandas as pd
from data_loader import load_financial_data

def inspect_dataset(df: pd.DataFrame):
    """
    Run initial exploratory checks on the dataset.
    """
    print("\n--- DATASET OVERVIEW ---")
    print(f"Total Rows: {df.shape[0]}")
    print(f"Total Columns: {df.shape[1]}")
    
    # Check key columns related to location and financial inclusion if they exist
    key_cols = ['county', 'Latitude', 'Longitude', 'PWD']
    existing_cols = [col for col in key_cols if col in df.columns]
    
    print("\n--- SAMPLE KEY COLUMNS SUMMARY ---")
    print(df[existing_cols].describe())

if __name__ == "__main__":
    df = load_financial_data("data/2024_Finaccess_Publicdata.dta")
    inspect_dataset(df)