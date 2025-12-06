
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Bank Term Deposit Prediction",
    page_icon="🏦",
    layout="wide"
)

# Title and Description
st.title("🏦 Bank Term Deposit Prediction App")
st.markdown("""
This app predicts whether a customer will subscribe to a term deposit based on their profile and marketing interactions.
""")

# Load Artifacts
@st.cache_resource
def load_artifacts():
    try:
        preprocessor = joblib.load('preprocessor.pkl')
        pca = joblib.load('pca.pkl')
        model = joblib.load('model.pkl')
        columns_info = joblib.load('columns_info.pkl')
        return preprocessor, pca, model, columns_info
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'save_model.py' has been run successfully.")
        return None, None, None, None

preprocessor, pca, model, columns_info = load_artifacts()

if preprocessor and model:
    # Sidebar for User Input
    st.sidebar.header("Customer Information")

    # Helper function to get categories from OneHotEncoder
    def get_categories(preprocessor, cat_feature_names):
        # Access the 'cat' transformer within ColumnTransformer
        # Note: In save_model.py we named it 'cat'
        try:
            # transformers_ returns a list of (name, transformer, columns)
            # We look for the one named 'cat'
            for name, trans, cols in preprocessor.transformers_:
                if name == 'cat':
                    return {col: cats for col, cats in zip(cat_feature_names, trans.categories_)}
        except Exception as e:
            st.warning(f"Could not automatically retrieve categories: {e}. Using defaults.")
            return {}

    cat_cols = columns_info['categorical']
    print(cat_cols)
    num_cols = columns_info['numerical']
    
    categories_map = get_categories(preprocessor, cat_cols)

    input_data = {}

    # --- Categorical Inputs ---
    st.sidebar.subheader("Demographics & Profile")
    
    # Define default options if extraction fails or as fallbacks
    defaults = {
        'job': ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'],
        'marital': ['divorced', 'married', 'single', 'unknown'],
        'education': ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'],
        'default': ['no', 'yes', 'unknown'],
        'housing': ['no', 'yes', 'unknown'],
        'loan': ['no', 'yes', 'unknown'],
        'contact': ['cellular', 'telephone'],
        'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
        'day_of_week': ['mon', 'tue', 'wed', 'thu', 'fri'],
        'poutcome': ['failure', 'nonexistent', 'success']
    }

    for col in cat_cols:
        options = categories_map.get(col, defaults.get(col, []))
        # Ensure options are list
        if isinstance(options, np.ndarray):
            options = options.tolist()
        
        input_data[col] = st.sidebar.selectbox(f"Select {col.capitalize()}", options)

    # --- Numerical Inputs ---
    st.sidebar.subheader("Campaign & Economic Indicators")
    
    # Age
    input_data['age'] = st.sidebar.slider("Age", 18, 100, 30)
    
    # Campaign Info
    input_data['campaign'] = st.sidebar.number_input("Number of contacts during this campaign", min_value=1, value=1)
    input_data['previous'] = st.sidebar.number_input("Number of contacts before this campaign", min_value=0, value=0)
    
    # Derived feature logic: p_contacted
    pdays = st.sidebar.number_input("Days since last contact (999 means never contacted)", min_value=0, value=999)
    input_data['p_contacted'] = 0 if pdays == 999 else 1
    
    # Economic Indicators (Default values roughly means from dataset)
    input_data['emp.var.rate'] = st.sidebar.number_input("Employment Variation Rate", value=-1.8)
    input_data['cons.price.idx'] = st.sidebar.number_input("Consumer Price Index", value=92.89)
    input_data['cons.conf.idx'] = st.sidebar.number_input("Consumer Confidence Index", value=-46.2)
    input_data['euribor3m'] = st.sidebar.number_input("Euribor 3 Month Rate", value=1.29)
    input_data['nr.employed'] = st.sidebar.number_input("Number of Employees", value=5099.1)

    # Create DataFrame
    input_df = pd.DataFrame([input_data])
    
    # Display Input Data
    with st.expander("View Input Data"):
        st.dataframe(input_df)

    # --- Prediction ---
    if st.button("Predict Subscription", type="primary"):
        try:
            # 1. Preprocess
            # Ensure columns order matches what ColumnTransformer expects
            # It selects columns by name, so order in df shouldn't strictly matter for fit_transform but transform needs consistency?
            # Creating df with all expected columns
            
            # Note: We need to handle the case where user input might trigger unseen categories?
            # OneHotEncoder was set with handle_unknown='ignore', so it should be fine.
            
            # Transform
            X_processed = preprocessor.transform(input_df)
            
            # 2. PCA
            X_pca = pca.transform(X_processed)
            
            # 3. Predict
            prediction = model.predict(X_pca)[0]
            probability = model.predict_proba(X_pca)[0][1]

            st.markdown("---")
            st.header("Prediction Result")
            
            if prediction == 1:
                st.success(f"**YES**, the customer is likely to subscribe to a term deposit.")
                st.metric(label="Confidence", value=f"{probability:.2%}")
            else:
                st.warning(f"**NO**, the customer is unlikely to subscribe.")
                st.metric(label="Probability of Yes", value=f"{probability:.2%}")
                
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.code(str(e))

else:
    st.info("Please wait for the model artifacts to be loaded.")
