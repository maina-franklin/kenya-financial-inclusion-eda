import pandas as pd

def clean_and_encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess and clean FinAccess data.
    """
    df_clean = df.copy()
    
    # Run dataset checks / print summary
    print(f"Dataset Rows: {df_clean.shape[0]}, Columns: {df_clean.shape[1]}")
    
    # Check key columns if present
    key_cols = ['county', 'Latitude', 'Longitude', 'PWD']
    existing_cols = [col for col in key_cols if col in df_clean.columns]
    if existing_cols:
        print("Summary of key columns:")
        print(df_clean[existing_cols].describe())
        
    return df_clean

if __name__ == "__main__":
    from src.data_loader import load_financial_data
    df = load_financial_data("data/2024_Finaccess_Publicdata.dta")
    clean_and_encode_data(df)