import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_credit_prediction_model(df: pd.DataFrame):
    """
    Trains a baseline Logistic Regression model to predict financial inclusion targets
    based on available numeric survey features.
    """
    print("🤖 Step 4: Training Machine Learning Classification Model...")
    
    # Select clean numeric features
    numeric_df = df.select_dtypes(include=['number']).dropna(axis=1, how='all')
    
    if numeric_df.shape[1] < 3:
        print("   Not enough numeric features for modeling.")
        return None
    
    target_col = numeric_df.columns[-1]
    feature_cols = [col for col in numeric_df.columns[:10] if col != target_col]
    
    if len(feature_cols) == 0:
        print("   Insufficient features for training matrix.")
        return None
        
    model_df = numeric_df[feature_cols + [target_col]].dropna()
    
    if len(model_df) < 100:
        print("   Dataset too small after dropping NaNs for modeling.")
        return None
        
    # Create binary target via median split for classification
    median_val = model_df[target_col].median()
    y = (model_df[target_col] > median_val).astype(int)
    X = model_df[feature_cols]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"   ✨ Model training complete! Baseline Logistic Regression Accuracy: {acc * 100:.2f}%")
    return model

if __name__ == "__main__":
    from src.data_loader import load_financial_data
    df = load_financial_data("data/2024_Finaccess_Publicdata.dta")
    train_credit_prediction_model(df)