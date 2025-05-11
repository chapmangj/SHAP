import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler 
import shap
import matplotlib.pyplot as plt

# Config
CSV_FILE_PATH = 'data.csv' # add your fi
SAMPLE_ID_COL = 'SampleID'  # Or None if not available/needed for output
ELEMENT_COLS = [
    'Ag_ppm','Al_pct','As_ppm','Au_ppm','Ba_ppm','Be_ppm','Bi_ppm','Ca_pct',
    'Cd_ppm','Ce_ppm','Co_ppm','Cr_ppm','Cs_ppm','Fe_pct','Ga_ppm',
    'Ge_ppm','Hf_ppm','In_ppm','K_pct','La_ppm','Li_ppm','Mg_pct','Mn_ppm',
    'Mo_ppm','Na_pct','Nb_ppm','Ni_ppm','P_ppm','Pb_ppm','Rb_ppm','Re_ppm',
    'S_pct','Sb_ppm','Sc_ppm','Se_ppm','Sn_ppm','Sr_ppm','Ta_ppm','Te_ppm',
    'Th_ppm','Ti_pct','Tl_ppm','U_ppm','V_ppm','W_ppm','Y_ppm','Zn_ppm','Zr_ppm'
]
DOMAIN_COLUMN = 'cluster' # Column defining the domains (e.g., 'Lithology')
TARGET_COLUMN_FOR_SHAP = 'Cu_ppm' #  Numeric column to predict and explain 

MODEL_TYPE_FOR_SHAP = 'RandomForestRegressor' # Options: 'RandomForestRegressor', 'GradientBoostingRegressor', 'LinearRegression'
SHAP_MAX_DISPLAY_FEATURES = 15
MIN_SAMPLES_PER_DOMAIN = 20 # Minimum samples required in a domain to perform analysis

# Helper

def load_and_prepare_data(filepath, id_col, element_cols, domain_col, target_col):
    """Loads data and checks for essential columns."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data with {df.shape[0]} samples and {df.shape[1]} columns.")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

    required_cols = element_cols + [domain_col, target_col]
    if id_col:
        required_cols.append(id_col)
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: The following required columns are missing from the CSV: {missing_cols}")
        return None

    # Ensure target column is numeric and convert
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        print(f"Warning: Target column '{target_col}' is not numeric. Attempting conversion...")
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        if df[target_col].isnull().all():
            print(f"Error: Target column '{target_col}' could not be converted to numeric or all values are NaN after conversion.")
            return None
        print(f"Target column '{target_col}' converted. {df[target_col].isnull().sum()} NaNs introduced.")

    # Ensure element columns are numeric
    for col in element_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: Element column '{col}' is not numeric. Attempting conversion...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            print(f"Element column '{col}' converted. {df[col].isnull().sum()} NaNs introduced if any.")
            
    return df

def preprocess_domain_data(df_domain_subset, element_cols, target_col):
    """Prepares X and y for a specific domain, handles NaNs."""
    X_domain = df_domain_subset[element_cols].copy()
    y_domain = df_domain_subset[target_col].copy()

    # Impute NaNs in features (X)
    if X_domain.isnull().sum().sum() > 0:
        print(f"  Imputing NaNs in feature (element) columns for the current domain ({X_domain.isnull().sum().sum()} total NaNs)...")
        imputer_X = SimpleImputer(strategy='median')
        X_domain_imputed = imputer_X.fit_transform(X_domain)
        X_domain = pd.DataFrame(X_domain_imputed, columns=X_domain.columns, index=X_domain.index)

    # Drop NaNs in target 
    if y_domain.isnull().sum() > 0:
        print(f"  Warning: {y_domain.isnull().sum()} NaNs found in target column '{target_col}' for the current domain.")
        #  Drop rows with NaN 
        valid_indices = y_domain.dropna().index
        X_domain = X_domain.loc[valid_indices]
        y_domain = y_domain.loc[valid_indices]
        print(f"  Dropped rows with NaN target. {len(y_domain)} samples remain for this domain's target.")
        if y_domain.empty:
            return None, None

        
    if X_domain.empty or y_domain.empty:
        print("  Error: After preprocessing, no data remains for this domain.")
        return None, None
        
    return X_domain, y_domain

def train_model_for_domain(X_train, y_train, model_type='RandomForestRegressor'):
    # Trains your model
    print(f"  Training {model_type} for the current domain...")
    if model_type == 'RandomForestRegressor':
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'GradientBoostingRegressor':
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == 'LinearRegression':

        model = LinearRegression()
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
    
    model.fit(X_train, y_train)
    print("  Model training complete.")
    return model #

def explain_domain_with_shap(model, X_data, feature_names, domain_value_name, 
                             sample_ids=None, original_df_domain=None, target_column_name=None):
    # Generates SHAP explanations and plots for a domain
    print(f"  Generating SHAP explanations for domain: {domain_value_name}...")
    

    if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        explainer = shap.TreeExplainer(model, data=X_data) # data can be background or X_data itself
    else: # For LinearRegression or other models
        background_data = shap.sample(X_data, min(100, X_data.shape[0]))
        explainer = shap.KernelExplainer(model.predict, background_data)

    shap_values = explainer.shap_values(X_data) # For some explainers, this might be explainer(X_data).values

    if isinstance(shap_values, list): # For multi-output models, SHAP values can be a list
        print("  Warning: SHAP values are a list, likely a multi-output model. Taking first output's SHAP values.")
        shap_values = shap_values[0]


    # Create a DataFrame for SHAP values
    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_data.index)
    if sample_ids is not None and SAMPLE_ID_COL is not None: # Add SampleID if available
        shap_df.insert(0, SAMPLE_ID_COL, sample_ids)
    
    # Add original target values to SHAP DataFrame for context
    if original_df_domain is not None and target_column_name is not None and target_column_name in original_df_domain.columns:
        shap_df[f'Original_{target_column_name}'] = original_df_domain.loc[X_data.index, target_column_name]
        # Add predicted target values
        shap_df[f'Predicted_{target_column_name}'] = model.predict(X_data)


    # --- SHAP Summary Plot (Bar) ---
    plt.figure(figsize=(8, max(6, len(feature_names)*0.3))) # Adjust figure size
    shap.summary_plot(shap_values, X_data, plot_type="bar", feature_names=feature_names, 
                      show=False, max_display=SHAP_MAX_DISPLAY_FEATURES)
    plt.title(f'SHAP Feature Importance for Domain: {domain_value_name}\n(Target: {TARGET_COLUMN_FOR_SHAP})')
    plt.tight_layout()
    plt.savefig(f"shap_summary_bar_{domain_value_name}.png")
    plt.show()

    # --- SHAP Summary Plot (Beeswarm/Dot) ---
    plt.figure(figsize=(10, max(6, len(feature_names)*0.3)))
    shap.summary_plot(shap_values, X_data, feature_names=feature_names, 
                      show=False, max_display=SHAP_MAX_DISPLAY_FEATURES) # Default is beeswarm
    plt.title(f'SHAP Summary Plot for Domain: {domain_value_name}\n(Target: {TARGET_COLUMN_FOR_SHAP})')
    plt.tight_layout()
    plt.savefig(f"shap_summary_beeswarm_{domain_value_name}.png")
    plt.show()
    
    print(f"  SHAP plots saved for domain: {domain_value_name}")
    return shap_df

def main():
    print("--- Domain-Specific SHAP Analysis ---")
    
    full_df = load_and_prepare_data(CSV_FILE_PATH, SAMPLE_ID_COL, ELEMENT_COLS, DOMAIN_COLUMN, TARGET_COLUMN_FOR_SHAP)
    if full_df is None:
        return

    if DOMAIN_COLUMN not in full_df.columns:
        print(f"Error: Specified DOMAIN_COLUMN '{DOMAIN_COLUMN}' not found in the dataset.")
        return
        
    unique_domains = full_df[DOMAIN_COLUMN].unique()
    print(f"\nFound {len(unique_domains)} unique domains in column '{DOMAIN_COLUMN}': {list(unique_domains)}")

    all_domain_shap_dfs = {}

    for domain_value in unique_domains:
        print(f"\n--- Processing Domain: {domain_value} ---")
        df_current_domain = full_df[full_df[DOMAIN_COLUMN] == domain_value].copy()

        if len(df_current_domain) < MIN_SAMPLES_PER_DOMAIN:
            print(f"  Skipping domain '{domain_value}' due to insufficient samples ({len(df_current_domain)} found, {MIN_SAMPLES_PER_DOMAIN} required).")
            continue
        
        print(f"  Number of samples in this domain: {len(df_current_domain)}")

        X_domain, y_domain = preprocess_domain_data(df_current_domain, ELEMENT_COLS, TARGET_COLUMN_FOR_SHAP)
        
        if X_domain is None or y_domain is None or X_domain.empty or y_domain.empty:
            print(f"  Skipping domain '{domain_value}' due to data preprocessing issues (e.g., all NaNs in target).")
            continue
        
        # For SHAP, we typically want to explain the model's behavior on the data it was trained on,
        model_domain = train_model_for_domain(X_domain, y_domain, MODEL_TYPE_FOR_SHAP)
        
        sample_ids_for_domain = df_current_domain.loc[X_domain.index, SAMPLE_ID_COL] if SAMPLE_ID_COL else None
        
        shap_df_domain = explain_domain_with_shap(model_domain, X_domain, ELEMENT_COLS, domain_value,
                                                  sample_ids=sample_ids_for_domain,
                                                  original_df_domain=df_current_domain,
                                                  target_column_name=TARGET_COLUMN_FOR_SHAP)
        
        if shap_df_domain is not None:
            try:
                csv_filename = f"shap_values_domain_{str(domain_value).replace(' ','_').lower()}.csv"
                shap_df_domain.to_csv(csv_filename)
                print(f"  SHAP values for domain '{domain_value}' saved to '{csv_filename}'")
                all_domain_shap_dfs[domain_value] = shap_df_domain
            except Exception as e:
                print(f"  Error saving SHAP values CSV for domain '{domain_value}': {e}")
                
    print("\n--- All Domain Processing Complete ---")

if __name__ == "__main__":
    main()
