import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import load_financial_data

def run_county_eda(df: pd.DataFrame):
    """
    Analyze survey distribution and metrics across Kenyan counties.
    """
    print("\n--- COUNTY DISTRIBUTION SUMMARY ---")
    county_counts = df['county'].value_counts().reset_index()
    county_counts.columns = ['county_code', 'respondent_count']
    print(county_counts.head(10))
    
    # Visualization: Top 10 counties by respondent count
    plt.figure(figsize=(10, 6))
    top_counties = county_counts.head(10)
    sns.barplot(
        data=top_counties, 
        x='county_code', 
        y='respondent_count', 
        hue='county_code', 
        palette='viridis', 
        legend=False
    )
    plt.title('Top 10 Counties by Survey Respondent Count')
    plt.xlabel('County Code')
    plt.ylabel('Number of Respondents')
    plt.tight_layout()
    
    # Save figure into the outputs folder
    plt.savefig('outputs/county_distribution.png')
    print("\nSaved chart visualization to outputs/county_distribution.png")

if __name__ == "__main__":
    df = load_financial_data("data/2024_Finaccess_Publicdata.dta")
    run_county_eda(df)