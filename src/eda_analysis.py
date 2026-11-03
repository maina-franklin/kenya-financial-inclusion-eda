import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from src.data_loader import load_financial_data

def generate_county_distribution(df: pd.DataFrame):
    """Generates and saves the county distribution bar chart."""
    print("   Generating county distribution chart...")
    plt.figure(figsize=(10, 6))
    
    county_col = next((col for col in df.columns if col.lower() in ['county', 'county_code', 'countycode']), None)
    
    if county_col:
        top_counties = df[county_col].value_counts().head(10)
        top_counties.plot(kind='bar', color='#2b5c8f', edgecolor='black')
        plt.title('Top 10 Counties by Survey Respondent Count', fontsize=14, fontweight='bold')
        plt.xlabel('County Code / Name', fontsize=12)
        plt.ylabel('Number of Respondents', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
    else:
        plt.text(0.5, 0.5, 'County Data Column Not Found', horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.title('County Distribution Placeholder')
        
    output_path = Path("outputs/county_distribution.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"   Saved {output_path}")

def generate_demographic_breakdown(df: pd.DataFrame):
    """Generates and saves a demographic vs financial access breakdown chart."""
    print("   Generating demographic breakdown analysis...")
    plt.figure(figsize=(10, 6))
    
    demo_col = next((col for col in df.columns if any(term in col.lower() for term in ['education', 'educ', 'age_group', 'gender', 'sex'])), None)
    
    if demo_col:
        value_counts = df[demo_col].value_counts().head(8)
        value_counts.plot(kind='barh', color='#3b82f6', edgecolor='black')
        plt.title(f'Demographic Breakdown: Top Categories in {demo_col}', fontsize=14, fontweight='bold')
        plt.xlabel('Respondent Count', fontsize=12)
        plt.ylabel('Category', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()
    else:
        top_cats = df.iloc[:, 2].value_counts().head(8)
        top_cats.plot(kind='barh', color='#10b981', edgecolor='black')
        plt.title('Demographic Distribution Overview', fontsize=14, fontweight='bold')
        plt.xlabel('Count', fontsize=12)
        plt.gca().invert_yaxis()
        plt.tight_layout()

    output_path = Path("outputs/demographic_breakdown.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"   Saved {output_path}")

def generate_wealth_credit_heatmap(df: pd.DataFrame):
    """Generates and saves the wealth vs credit heatmap."""
    print("   Generating wealth vs credit heatmap...")
    plt.figure(figsize=(8, 6))
    
    numeric_df = df.select_dtypes(include=['number'])
    if not numeric_df.empty and numeric_df.shape[1] > 1:
        corr = numeric_df.iloc[:, :5].corr()
        sns.heatmap(corr, annot=True, cmap='Blues', fmt=".2f", linewidths=0.5)
        plt.title('Wealth vs Financial Indicators Heatmap', fontsize=14, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'Insufficient Numeric Data for Heatmap', horizontalalignment='center', verticalalignment='center', fontsize=14)
        plt.title('Wealth vs Credit Heatmap Placeholder')
        
    output_path = Path("outputs/wealth_vs_credit_heatmap.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"   Saved {output_path}")