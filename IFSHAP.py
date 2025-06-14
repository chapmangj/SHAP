import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import MinMaxScaler
import shap
import matplotlib.pyplot as plt

# --- General Configuration ---
CSV_FILE_PATH = 'rf_clustered_drillhole_data.csv' # <<< SET YOUR CSV FILE PATH
SAMPLE_ID_COL = 'SampleID'  # Or None if not used

ELEMENT_COLS = [ # <<< List all your element columns (these are potential features)
    'Ag_ppm','Al_pct','As_ppm','Au_ppm','Ba_ppm','Be_ppm','Bi_ppm','Ca_pct',
    'Cd_ppm','Ce_ppm','Co_ppm','Cr_ppm','Cs_ppm','Fe_pct','Ga_ppm', 'Cu_ppm',
    'Ge_ppm','Hf_ppm','In_ppm','K_pct','La_ppm','Li_ppm','Mg_pct','Mn_ppm',
    'Mo_ppm','Na_pct','Nb_ppm','Ni_ppm','P_ppm','Pb_ppm','Rb_ppm','Re_ppm',
    'S_pct','Sb_ppm','Sc_ppm','Se_ppm','Sn_ppm','Sr_ppm','Ta_ppm','Te_ppm',
    'Th_ppm','Ti_pct','Tl_ppm','U_ppm','V_ppm','W_ppm','Y_ppm','Zn_ppm','Zr_ppm'
]
SHAP_MAX_DISPLAY_FEATURES = 15

# --- CHOOSE ANALYSIS TYPE ---
ANALYSIS_TYPE = "REGRESSION"  # Options: "BINARY_CLASSIFICATION", "REGRESSION"
# ANALYSIS_TYPE = "REGRESSION"

# --- Configuration for BINARY CLASSIFICATION 
BINARY_TARGET_COLUMN = 'cluster'        # Column with categories (e.g., cluster, lithology)
BINARY_POSITIVE_CLASS_VALUE = '5'      # The specific category value to be class 1 (will be treated as string)
BINARY_CLASSIFIER_MODEL_TYPE = 'RandomForestClassifier' # e.g., 'RandomForestClassifier', 'GradientBoostingClassifier', 'LogisticRegression'
MIN_SAMPLES_PER_CLASS_BINARY = 10

# --- Configuration for REGRESSION
REGRESSION_TARGET_ELEMENT = 'Cu_ppm'     # Element to predict
REGRESSION_MODEL_TYPE = 'RandomForestRegressor' # e.g., 'GradientBoostingRegressor', 'RandomForestRegressor', 'LinearRegression'
# To run regression for each domain separately, specify the domain column:
REGRESSION_DOMAIN_COLUMN = 'cluster' # e.g., 'cluster1', 'lith'. Set to None or "" to run globally on all data.
MIN_SAMPLES_FOR_REGRESSION_DOMAIN = 15 # Min samples needed if running per-domain regression


# --- Helper Functions ---

def load_and_prepare_data(filepath, id_col, element_cols,
                           analysis_type, binary_target_col=None, regression_target_el=None, regression_domain_col=None):
    """Loads data and checks for essential columns based on analysis type."""
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded data with {df.shape[0]} samples and {df.shape[1]} columns.")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

    if not element_cols:
        print("Error: ELEMENT_COLS is empty. Please specify feature columns.")
        return None

    required_cols = list(element_cols)
    if id_col:
        required_cols.append(id_col)

    if analysis_type == "BINARY_CLASSIFICATION":
        if not binary_target_col:
            print("Error: BINARY_TARGET_COLUMN must be specified for BINARY_CLASSIFICATION.")
            return None
        required_cols.append(binary_target_col)
    elif analysis_type == "REGRESSION":
        if not regression_target_el:
            print("Error: REGRESSION_TARGET_ELEMENT must be specified for REGRESSION.")
            return None
        if regression_target_el not in df.columns: # Check if target element exists
             print(f"Error: REGRESSION_TARGET_ELEMENT '{regression_target_el}' not found in CSV.")
             return None
        if regression_target_el not in required_cols: # Add if not already in (e.g. if it's also a feature)
            required_cols.append(regression_target_el)

        if regression_domain_col and regression_domain_col not in df.columns:
            print(f"Error: REGRESSION_DOMAIN_COLUMN '{regression_domain_col}' not found in CSV.")
            return None
        if regression_domain_col and regression_domain_col not in required_cols:
            required_cols.append(regression_domain_col)
    else:
        print(f"Error: Invalid ANALYSIS_TYPE: {analysis_type}")
        return None

    missing_cols = [col for col in required_cols if col not in df.columns and col is not None] # Ensure col is not None
    if missing_cols:
        print(f"Error: The following required columns are missing from the CSV: {missing_cols}")
        return None

    # Ensure element columns (features) are numeric
    for col in element_cols:
        if col not in df.columns:
            print(f"Error: Feature column '{col}' from ELEMENT_COLS not found in CSV.")
            return None
        if not pd.api.types.is_numeric_dtype(df[col]):
            print(f"Warning: Feature column '{col}' is not numeric. Attempting conversion...")
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().all():
                 print(f"Warning: Feature column '{col}' became all NaNs after conversion.")
            else:
                print(f"Feature column '{col}' converted. {df[col].isnull().sum()} NaNs introduced if any.")
    
    # For regression, ensure target element is numeric
    if analysis_type == "REGRESSION" and regression_target_el in df.columns:
        if not pd.api.types.is_numeric_dtype(df[regression_target_el]):
            print(f"Warning: Regression target '{regression_target_el}' is not numeric. Attempting conversion...")
            df[regression_target_el] = pd.to_numeric(df[regression_target_el], errors='coerce')
            if df[regression_target_el].isnull().all():
                print(f"Error: Regression target '{regression_target_el}' is all NaNs after conversion.")
                return None
    return df

def preprocess_features_log_impute_scale(X_df):
    """Applies log1p, imputation, and MinMax scaling to features (X)."""
    X_transformed = X_df.copy()
    print("  Preprocessing features (X):")
    print("    Applying log1p (log(x+1)) transformation...")
    for col in X_transformed.columns:
        if pd.api.types.is_numeric_dtype(X_transformed[col]):
            if (X_transformed[col] < 0).any():
                neg_count = (X_transformed[col] < 0).sum()
                print(f"      Warning: Column {col} contains {neg_count} negative values. Clipping to 0 before log1p.")
                X_transformed[col] = X_transformed[col].clip(lower=0)
            X_transformed[col] = np.log1p(X_transformed[col].astype(float))
        else:
            print(f"      Warning: Column {col} is not numeric, skipping log transform for it.")

    if X_transformed.isnull().sum().sum() > 0:
        print(f"    Imputing NaNs in log-transformed features ({X_transformed.isnull().sum().sum()} total NaNs using median)...")
        imputer_X = SimpleImputer(strategy='median')
        X_imputed = imputer_X.fit_transform(X_transformed)
        X_transformed = pd.DataFrame(X_imputed, columns=X_transformed.columns, index=X_transformed.index)
    else:
        print("    No NaNs found in features after log transform.")

    print("    Scaling log-transformed, imputed features to 0-1 range (MinMaxScaler)...")
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_transformed)
    X_transformed = pd.DataFrame(X_scaled, columns=X_transformed.columns, index=X_transformed.index)
    print("  Feature (X) preprocessing complete.")
    return X_transformed

# --- Binary Classification Functions ---
def preprocess_for_binary_classification(df, feature_element_cols, target_col, positive_class_val):
    df[target_col] = df[target_col].astype(str)
    positive_class_val_str = str(positive_class_val)

    if positive_class_val_str not in df[target_col].unique():
        print(f"Error: Positive class value '{positive_class_val_str}' not found in target column '{target_col}'.")
        print(f"       Available unique values: {df[target_col].unique()}")
        return None, None

    X_raw = df[feature_element_cols].copy()
    y = (df[target_col] == positive_class_val_str).astype(int)

    X = preprocess_features_log_impute_scale(X_raw)
    
    common_index = X.index.intersection(y.index) # Align X and y after potential row drops in X or y
    X = X.loc[common_index]
    y = y.loc[common_index]

    if X.empty or y.empty: return None, None
    if len(y.unique()) < 2:
        print(f"  Error: Binary target for class '{positive_class_val_str}' has only one class. Cannot train classifier.")
        return None, None
    class_counts = y.value_counts()
    if class_counts.min() < MIN_SAMPLES_PER_CLASS_BINARY:
        print(f"  Warning: Insufficient samples in one class. Class 0: {class_counts.get(0,0)}, Class 1: {class_counts.get(1,0)}. Min required: {MIN_SAMPLES_PER_CLASS_BINARY}")
    return X, y

def train_binary_classifier(X_train, y_train, model_type):
    print(f"  Training {model_type} for binary classification...")
    if model_type == 'RandomForestClassifier': model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
    elif model_type == 'GradientBoostingClassifier': model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    elif model_type == 'LogisticRegression': model = LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced')
    else: raise ValueError(f"Unsupported binary classifier: {model_type}")
    model.fit(X_train, y_train)
    return model

def explain_binary_classifier_with_shap(model, X_data, feature_names, positive_class_name,
                                        original_df_for_context=None, sample_id_col=None, binary_target_col=None):
    print(f"  Generating SHAP for binary classification (Target: Is '{positive_class_name}'?)...")
    if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier)):
        explainer = shap.TreeExplainer(model, data=X_data) 
        shap_values_raw = explainer.shap_values(X_data)
    else: 
        background_data = shap.sample(X_data, min(100, X_data.shape[0])) 
        explainer = shap.KernelExplainer(model.predict_proba, background_data) # Use predict_proba for classifiers
        shap_values_raw = explainer.shap_values(X_data)

    if isinstance(shap_values_raw, list) and len(shap_values_raw) == 2: shap_values_for_positive_class = shap_values_raw[1] 
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 3 and shap_values_raw.shape[2] == 2: shap_values_for_positive_class = shap_values_raw[:, :, 1]
    elif isinstance(shap_values_raw, np.ndarray) and shap_values_raw.ndim == 2: shap_values_for_positive_class = shap_values_raw
    else: print("  Warning: SHAP values format not standard. Using raw."); shap_values_for_positive_class = shap_values_raw

    shap_df = pd.DataFrame(shap_values_for_positive_class, columns=feature_names, index=X_data.index)
    
    if original_df_for_context is not None:
        aligned_original_df = original_df_for_context.loc[X_data.index]
        if sample_id_col and sample_id_col in aligned_original_df.columns:
            shap_df.insert(0, sample_id_col, aligned_original_df[sample_id_col])
        if binary_target_col:
            shap_df[f'Actual_Is_{str(positive_class_name).replace(" ","_")}'] = (aligned_original_df[binary_target_col].astype(str) == str(positive_class_name)).astype(int)
        shap_df[f'Predicted_Prob_Is_{str(positive_class_name).replace(" ","_")}'] = model.predict_proba(X_data)[:, 1]

    plt.figure(figsize=(8, max(6, len(feature_names)*0.3)))
    shap.summary_plot(shap_values_for_positive_class, X_data, plot_type="bar", feature_names=feature_names, show=False, max_display=SHAP_MAX_DISPLAY_FEATURES)
    plt.title(f'SHAP Importance: Is "{positive_class_name}"?\n(Features Log+Scaled)')
    plt.tight_layout(); plt.savefig(f"shap_bar_binary_{str(positive_class_name).replace(' ','_').replace('/','_')}.png"); plt.show()
    plt.figure(figsize=(10, max(6, len(feature_names)*0.3)))
    shap.summary_plot(shap_values_for_positive_class, X_data, feature_names=feature_names, show=False, max_display=SHAP_MAX_DISPLAY_FEATURES)
    plt.title(f'SHAP Summary: Is "{positive_class_name}"?\n(Features Log+Scaled)')
    plt.tight_layout(); plt.savefig(f"shap_beeswarm_binary_{str(positive_class_name).replace(' ','_').replace('/','_')}.png"); plt.show()
    return shap_df

# --- Regression Functions ---
def preprocess_for_regression(df_subset, all_element_cols, target_element_col):
    """Prepares X (transformed features) and y (target element) for regression."""
    if target_element_col not in df_subset.columns:
        print(f"  Error: Target element '{target_element_col}' not found in this data subset.")
        return None, None
    
    y = df_subset[target_element_col].copy()
    
    # Features are all other element columns
    feature_cols_for_regression = [col for col in all_element_cols if col != target_element_col]
    if not feature_cols_for_regression:
        print("  Error: No feature columns left after excluding the target element for regression.")
        return None, None
    X_raw = df_subset[feature_cols_for_regression].copy()

    # Preprocess features (X)
    X = preprocess_features_log_impute_scale(X_raw)

    # Handle NaNs in target (y) and align X
    if y.isnull().any():
        print(f"  {y.isnull().sum()} NaNs found in target '{target_element_col}'. Dropping corresponding rows.")
        valid_indices = y.dropna().index
        X = X.loc[X.index.intersection(valid_indices)]
        y = y.loc[valid_indices]
        if y.empty:
            print("  Error: Target column is empty after dropping NaNs.")
            return None, None
            
    # Align X and y again after all processing
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]

    if X.empty or y.empty:
        print("  Error: No data remains for regression after preprocessing X and y.")
        return None, None
    return X, y

def train_regression_model(X_train, y_train, model_type):
    print(f"  Training {model_type} for regression...")
    if model_type == 'RandomForestRegressor': model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == 'GradientBoostingRegressor': model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    elif model_type == 'LinearRegression': model = LinearRegression()
    else: raise ValueError(f"Unsupported regression model: {model_type}")
    model.fit(X_train, y_train)
    return model

def explain_regression_with_shap(model, X_data, feature_names, target_element_name, domain_identifier="Global",
                                 original_df_for_context=None, sample_id_col=None):
    print(f"  Generating SHAP for regression (Target: {target_element_name}, Domain: {domain_identifier})...")
    if isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
        explainer = shap.TreeExplainer(model, data=X_data)
    else: # For LinearRegression
        background_data = shap.sample(X_data, min(100, X_data.shape[0]))
        explainer = shap.KernelExplainer(model.predict, background_data)
    shap_values = explainer.shap_values(X_data)

    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_data.index)

    if original_df_for_context is not None:
        aligned_original_df = original_df_for_context.loc[X_data.index]
        if sample_id_col and sample_id_col in aligned_original_df.columns:
            shap_df.insert(0, sample_id_col, aligned_original_df[sample_id_col])
        # Add original target and predicted target
        shap_df[f'Original_{target_element_name}'] = aligned_original_df[target_element_name]
        shap_df[f'Predicted_{target_element_name}'] = model.predict(X_data)

    domain_tag = str(domain_identifier).replace(' ','_').replace('/','_')
    plt.figure(figsize=(8, max(6, len(feature_names)*0.3)))
    shap.summary_plot(shap_values, X_data, plot_type="bar", feature_names=feature_names, show=False, max_display=SHAP_MAX_DISPLAY_FEATURES)
    plt.title(f'SHAP Importance for {target_element_name} (Domain: {domain_identifier})\n(Features Log+Scaled)')
    plt.tight_layout(); plt.savefig(f"shap_bar_regr_{domain_tag}_{target_element_name}.png"); plt.show()
    plt.figure(figsize=(10, max(6, len(feature_names)*0.3)))
    shap.summary_plot(shap_values, X_data, feature_names=feature_names, show=False, max_display=SHAP_MAX_DISPLAY_FEATURES)
    plt.title(f'SHAP Summary for {target_element_name} (Domain: {domain_identifier})\n(Features Log+Scaled)')
    plt.tight_layout(); plt.savefig(f"shap_beeswarm_regr_{domain_tag}_{target_element_name}.png"); plt.show()
    return shap_df

# --- Main Execution ---
def main():
    print(f"--- Starting SHAP Analysis: Type = {ANALYSIS_TYPE} ---")
    full_df = load_and_prepare_data(CSV_FILE_PATH, SAMPLE_ID_COL, ELEMENT_COLS,
                                   ANALYSIS_TYPE, BINARY_TARGET_COLUMN,
                                   REGRESSION_TARGET_ELEMENT, REGRESSION_DOMAIN_COLUMN)
    if full_df is None:
        print("Failed to load or prepare data. Exiting.")
        return

    if ANALYSIS_TYPE == "BINARY_CLASSIFICATION":
        if not BINARY_TARGET_COLUMN or BINARY_POSITIVE_CLASS_VALUE is None: # Check if None explicitly for value
            print("Error: BINARY_TARGET_COLUMN and BINARY_POSITIVE_CLASS_VALUE must be set for binary classification. Exiting.")
            return
        
        print(f"\n--- Performing Binary Classification for Target Column: '{BINARY_TARGET_COLUMN}', Positive Class: '{BINARY_POSITIVE_CLASS_VALUE}' ---")
        X_binary, y_binary = preprocess_for_binary_classification(
            full_df.copy(), ELEMENT_COLS, BINARY_TARGET_COLUMN, BINARY_POSITIVE_CLASS_VALUE
        )
        if X_binary is not None and y_binary is not None:
            if len(y_binary.value_counts()) < 2 or y_binary.value_counts().min() < MIN_SAMPLES_PER_CLASS_BINARY:
                print(f"  Skipping: Insufficient samples or only one class present after preprocessing. Counts: {y_binary.value_counts().to_dict()}")
            else:
                print(f"  Data prepared: {X_binary.shape[0]} samples. Target distribution: {y_binary.value_counts().to_dict()}")
                model = train_binary_classifier(X_binary, y_binary, BINARY_CLASSIFIER_MODEL_TYPE)
                shap_df = explain_binary_classifier_with_shap(
                    model, X_binary, ELEMENT_COLS, BINARY_POSITIVE_CLASS_VALUE,
                    original_df_for_context=full_df.loc[X_binary.index], # Pass aligned original data
                    sample_id_col=SAMPLE_ID_COL, binary_target_col=BINARY_TARGET_COLUMN
                )
                if shap_df is not None:
                    csv_fn = f"shap_binary_{str(BINARY_POSITIVE_CLASS_VALUE).replace(' ','_').replace('/','_')}.csv"
                    shap_df.to_csv(csv_fn); print(f"  SHAP values saved to '{csv_fn}'")
        else:
            print("  Skipping binary classification due to preprocessing issues.")

    elif ANALYSIS_TYPE == "REGRESSION":
        if not REGRESSION_TARGET_ELEMENT:
            print("Error: REGRESSION_TARGET_ELEMENT must be set for regression. Exiting.")
            return
        if REGRESSION_TARGET_ELEMENT not in ELEMENT_COLS:
             print(f"Warning: REGRESSION_TARGET_ELEMENT '{REGRESSION_TARGET_ELEMENT}' is not in ELEMENT_COLS. It will only be a target, not a feature if used by mistake in ELEMENT_COLS.")
        
        feature_cols_for_regr = [col for col in ELEMENT_COLS if col != REGRESSION_TARGET_ELEMENT]
        if not feature_cols_for_regr:
            print("Error: No feature columns available for regression after excluding the target element. Exiting.")
            return

        if REGRESSION_DOMAIN_COLUMN and REGRESSION_DOMAIN_COLUMN in full_df.columns:
            print(f"\n--- Performing Per-Domain Regression for Target Element: '{REGRESSION_TARGET_ELEMENT}', Domain Column: '{REGRESSION_DOMAIN_COLUMN}' ---")
            unique_domains = full_df[REGRESSION_DOMAIN_COLUMN].dropna().unique()
            for domain_value in unique_domains:
                domain_value_str = str(domain_value) # Ensure string for filenames etc.
                print(f"\n  -- Processing Domain: {domain_value_str} --")
                df_domain_subset = full_df[full_df[REGRESSION_DOMAIN_COLUMN] == domain_value].copy()
                if len(df_domain_subset) < MIN_SAMPLES_FOR_REGRESSION_DOMAIN:
                    print(f"    Skipping domain '{domain_value_str}': Insufficient samples ({len(df_domain_subset)}). Min required: {MIN_SAMPLES_FOR_REGRESSION_DOMAIN}")
                    continue
                
                X_regr, y_regr = preprocess_for_regression(df_domain_subset, ELEMENT_COLS, REGRESSION_TARGET_ELEMENT)
                if X_regr is not None and y_regr is not None:
                    if len(X_regr) < MIN_SAMPLES_FOR_REGRESSION_DOMAIN : # Check again after NaN drops
                         print(f"    Skipping domain '{domain_value_str}': Insufficient samples after preprocessing ({len(X_regr)}).")
                         continue
                    print(f"    Data prepared: {X_regr.shape[0]} samples.")
                    model = train_regression_model(X_regr, y_regr, REGRESSION_MODEL_TYPE)
                    shap_df = explain_regression_with_shap(
                        model, X_regr, feature_cols_for_regr, REGRESSION_TARGET_ELEMENT, domain_identifier=domain_value_str,
                        original_df_for_context=df_domain_subset.loc[X_regr.index], sample_id_col=SAMPLE_ID_COL
                    )
                    if shap_df is not None:
                        csv_fn = f"shap_regr_domain_{domain_value_str.replace(' ','_').replace('/','_')}_{REGRESSION_TARGET_ELEMENT}.csv"
                        shap_df.to_csv(csv_fn); print(f"    SHAP values saved to '{csv_fn}'")
                else:
                    print(f"    Skipping domain '{domain_value_str}' due to preprocessing issues.")
        else:
            print(f"\n--- Performing Global Regression for Target Element: '{REGRESSION_TARGET_ELEMENT}' (All Data) ---")
            if REGRESSION_DOMAIN_COLUMN: # User specified a domain column but it wasn't found
                 print(f"  Warning: REGRESSION_DOMAIN_COLUMN '{REGRESSION_DOMAIN_COLUMN}' was specified but not found or is empty. Running globally.")

            X_regr, y_regr = preprocess_for_regression(full_df.copy(), ELEMENT_COLS, REGRESSION_TARGET_ELEMENT)
            if X_regr is not None and y_regr is not None:
                if len(X_regr) < MIN_SAMPLES_FOR_REGRESSION_DOMAIN : # Use a general threshold
                         print(f"  Skipping global regression: Insufficient samples after preprocessing ({len(X_regr)}).")
                else:
                    print(f"  Data prepared: {X_regr.shape[0]} samples.")
                    model = train_regression_model(X_regr, y_regr, REGRESSION_MODEL_TYPE)
                    shap_df = explain_regression_with_shap(
                        model, X_regr, feature_cols_for_regr, REGRESSION_TARGET_ELEMENT, domain_identifier="Global",
                        original_df_for_context=full_df.loc[X_regr.index], sample_id_col=SAMPLE_ID_COL
                    )
                    if shap_df is not None:
                        csv_fn = f"shap_regr_global_{REGRESSION_TARGET_ELEMENT}.csv"
                        shap_df.to_csv(csv_fn); print(f"  SHAP values saved to '{csv_fn}'")
            else:
                print("  Skipping global regression due to preprocessing issues.")
    else:
        print(f"Error: Unknown ANALYSIS_TYPE '{ANALYSIS_TYPE}'. Please choose 'BINARY_CLASSIFICATION' or 'REGRESSION'.")

    print("\n--- Analysis Complete ---")

if __name__ == "__main__":
    main()
