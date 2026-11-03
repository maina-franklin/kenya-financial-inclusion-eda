import pandas as pd
from pathlib import Path

def load_financial_data(file_path: str) -> pd.DataFrame:
    """
    Load the Kenya Financial Inclusion Stata (.dta) dataset.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Please place your .dta file inside the data/ folder.")
    
    # Disable automatic categorical conversion to bypass duplicate label errors
    df = pd.read_stata(path, convert_categoricals=False)
    print(f"Successfully loaded Stata dataset with shape: {df.shape}")
    return df

if __name__ == "__main__":
    DATA_PATH = "data/2024_Finaccess_Publicdata.dta"
    try:
        df = load_financial_data(DATA_PATH)
        print("\nDataset Head Preview:")
        print(df.head())
    except FileNotFoundError as e:
        print(e)