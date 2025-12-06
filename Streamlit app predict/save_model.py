
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def save_best_model():

    print("Loading data...")

    # Load dataset
    data_path = r"C:\Users\ADMIN\Desktop\bài ck nhưng ko up github\filedl\bank-additional-full.csv"
    try:
        df = pd.read_csv(data_path, delimiter=';')
    except FileNotFoundError:
        print(f"File not found at {data_path}, trying local directory...")
        df = pd.read_csv("bank-additional-full.csv", delimiter=';')


    print("Data loaded.")


    # --- Preprocessing Logic from Notebook ---
    
    # 1. Feature Engineering: p_contacted
    if 'pdays' in df.columns:
        df['p_contacted'] = [0 if x == 999 else 1 for x in df.pdays]
        df.drop(columns=['pdays'], inplace=True)
    
    # 2. Drop duration
    if 'duration' in df.columns:
        df = df.drop('duration', axis=1)
        
    # 3. Target Encoding
    if 'y' in df.columns:
        df['y_encoded'] = df['y'].map({'yes': 1, 'no': 0})
        df = df.drop('y', axis=1)

    # 4. Prepare X and y
    X = df.drop('y_encoded', axis=1)
    y = df['y_encoded']

    # 5. Split Train/Val/Test (Matching Notebook's 'Automatic Recovery' block)
    # Chia 85/15 trước
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    # Từ 85% còn lại, chia tiếp 80/20 cho Train/Val
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, random_state=42, stratify=y_temp)

    # 6. Define Features
    categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    numerical_features = ['emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'p_contacted', 'age', 'campaign', 'previous']

    # Ensure columns exist
    categorical_features = [c for c in categorical_features if c in X_train.columns]
    numerical_features = [c for c in numerical_features if c in X_train.columns]

    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")

    # 7. Create Preprocessor
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])


    print("Fitting Preprocessor...")

    X_train_proc = preprocessor.fit_transform(X_train)
    
    # 8. Create and Fit PCA

    print("Fitting PCA...")

    pca = PCA(n_components=0.95, random_state=42)
    X_train_pca = pca.fit_transform(X_train_proc)
    
    # 9. Apply SMOTE

    print("Applying SMOTE...")

    smote = SMOTE(sampling_strategy='minority', random_state=42)
    X_train_final, y_train_final = smote.fit_resample(X_train_pca, y_train)

    # 10. Train Random Forest (Best Model from Notebook)

    print("Training Best Model (Random Forest)...")

    # Params: {'n_estimators': 100, 'min_samples_leaf': 2, 'max_depth': None}
    model = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_depth=None, random_state=42, n_jobs=-1)
    model.fit(X_train_final, y_train_final)

    print("Model trained.")


    # 11. Save Artifacts

    print("Saving artifacts...")

    joblib.dump(preprocessor, 'preprocessor.pkl')
    joblib.dump(pca, 'pca.pkl')
    joblib.dump(model, 'model.pkl')
    
    # Also save the feature lists for the app to know inputs
    columns_info = {
        'categorical': categorical_features,
        'numerical': numerical_features
    }
    joblib.dump(columns_info, 'columns_info.pkl')


    print("All artifacts saved successfully!")


if __name__ == "__main__":
    save_best_model()
