"""
Main Orchestration Script for Kenya Financial Inclusion EDA (2024)
Author: Maina Franklin
"""

import os
import pandas as pd
from src.data_loader import load_financial_data
from src.preprocessing import clean_and_encode_data
from src.eda_analysis import generate_county_distribution, generate_demographic_breakdown, generate_wealth_credit_heatmap

def main():
    print("🚀 Starting Kenya FinAccess Data Pipeline...")
    
    # Define data path
    data_path = "data/2024_Finaccess_Publicdata.dta"
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset not found at {data_path}. Please ensure the .dta file is in the data/ folder.")
        return

    # Step 1: Ingest Data
    print("📥 Step 1: Loading raw dataset...")
    df_raw = load_financial_data(data_path)
    print(f"   Successfully loaded dataset with {df_raw.shape[0]:,} rows and {df_raw.shape[1]:,} columns.")

    # Step 2: Preprocess & Clean
    print("🧹 Step 2: Running preprocessing and encoding pipeline...")
    df_clean = clean_and_encode_data(df_raw)
    print("   Preprocessing complete.")

    # Step 3: Generate Visualizations & Outputs
    print("📊 Step 3: Generating analysis charts and heatmaps...")
    os.makedirs("outputs", exist_ok=True)
    
    generate_county_distribution(df_clean)
    generate_demographic_breakdown(df_clean)
    generate_wealth_credit_heatmap(df_clean)
    
    print("✨ Pipeline executed successfully! All outputs saved to /outputs/")

if __name__ == "__main__":
    main()