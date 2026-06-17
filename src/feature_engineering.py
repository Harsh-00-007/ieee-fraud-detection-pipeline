import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def select_features(df: pd.DataFrame, features_to_keep: list) -> pd.DataFrame:
    """Filters the dataframe to keep only specified features and core columns."""
    # We always want to keep TransactionID and isFraud (if it exists in the set)
    core_cols = ['TransactionID', 'isFraud']
    
    # Ensure we only try to keep columns that actually exist in the dataframe
    final_features = [col for col in features_to_keep if col in df.columns]
    
    # Add core columns if they aren't already in the list and exist in df
    for col in core_cols:
        if col in df.columns and col not in final_features:
            final_features.append(col)
            
    return df[final_features]

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna('missing')
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

            # STEP 2: Save the encoded data back to the dataframe as an integer
            df[col] = df[col].astype(int)
    return df

def run_feature_pipeline(df: pd.DataFrame, features_to_keep: list) -> pd.DataFrame:
    print("Starting feature pipeline...")
    df.columns = df.columns.str.replace('-', '_')
    # 1. Drop noise FIRST to save memory and processing time
    df = select_features(df, features_to_keep)
    print(f"Reduced dataset to {df.shape[1]} columns.")
    
    # 2. Clean the remaining features
    df = handle_missing_values(df)
    df = encode_categorical_features(df)
    
    print("Pipeline complete.")
    return df