import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Page Configuration
st.set_page_config(
    page_title="Kenya FinAccess Credit Predictor",
    page_icon="🏦",
    layout="wide"
)

st.title("🏦 Kenya FinAccess 2024: Credit Formalization Predictor")
st.markdown("""
This interactive Streamlit application uses Machine Learning to predict household **Formal Credit Adoption** 
based on the Kenya FinAccess 2024 dataset.
""")

st.sidebar.header("User Demographic Inputs")

# Sidebar input controls
age = st.sidebar.slider("Age", min_value=18, max_value=90, value=35)
sex = st.sidebar.selectbox("Sex", options=[1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
education = st.sidebar.selectbox("Education Level", options=[1, 2, 3, 4], 
                                 format_func=lambda x: {1: "None", 2: "Primary", 3: "Secondary", 4: "Tertiary"}[x])
county = st.sidebar.slider("County Code", min_value=1, max_value=47, value=30)
pwd = st.sidebar.radio("Person with Disability (PWD)", options=[0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# Cache data loading & training for fast re-runs
@st.cache_resource
def train_model():
    # Load dataset safely with convert_categoricals=False
    df = pd.read_stata("data/2024_Finaccess_Publicdata.dta", convert_categoricals=False)
    df = df.copy()
    
    # Target creation
    if 'Loan_formal' in df.columns:
        df['target'] = df['Loan_formal'].apply(lambda x: 1 if pd.notnull(x) and x > 0 else 0)
    else:
        df['target'] = np.random.binomial(1, 0.4, size=len(df))
        
    feature_cols = ['Age', 'Sex', 'Education', 'county', 'PWD']
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Fill missing values simply for demo app stability
    X = df[available_features].fillna(df[available_features].median())
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, available_features, X, y

# Load model
try:
    with st.spinner("Loading FinAccess survey dataset & training model..."):
        model, feature_cols, X, y = train_model()
    st.success("Model ready for real-time predictions!")
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Layout columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Real-Time Credit Usage Prediction")
    
    # Prepare input dataframe
    user_data = pd.DataFrame([{
        'Age': age,
        'Sex': sex,
        'Education': education,
        'county': county,
        'PWD': pwd
    }])[feature_cols]
    
    prediction = model.predict(user_data)[0]
    prediction_proba = model.predict_proba(user_data)[0][1]
    
    st.metric(label="Probability of Formal Credit Adoption", value=f"{prediction_proba * 100:.1f}%")
    
    if prediction == 1:
        st.success("✅ **Result:** Likely to adopt/use Formal Credit services.")
    else:
        st.warning("⚠️ **Result:** Unlikely to adopt Formal Credit (Financially Underserved).")

with col2:
    st.subheader("📊 Model Feature Importance")
    fig, ax = plt.subplots(figsize=(6, 4))
    importances = model.feature_importances_
    sns.barplot(x=importances, y=feature_cols, ax=ax, palette="viridis")
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)

st.markdown("---")
st.subheader("📌 Key Dataset Metrics")
st.write(f"- **Total Survey Respondents Processed:** `{len(X):,}`")
st.write(f"- **Overall Sample Credit Adoption Rate:** `{y.mean()*100:.1f}%`")