import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os
import pandas as pd

def prepare_data(train_df: pd.DataFrame):
    """Splits the processed training data into training and validation sets."""
    print("Splitting data into features (X) and target (y)...")
    
    # Separate the target variable ('isFraud') and drop IDs from features
    X = train_df.drop(['TransactionID', 'isFraud'], axis=1)
    y = train_df['isFraud']
    
    # 80/20 split: 80% to train, 20% to validate. 
    # stratify=y ensures the 20% validation set has the same ratio of fraud as the 80% set.
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_val, y_train, y_val

def train_xgboost(X_train, y_train, X_val, y_val):
    """Trains an XGBoost classifier with early stopping to prevent overfitting."""
    print("Initializing XGBoost model...")
    
    # SDE-Level Setup: We use 'auc' (Area Under the Curve) because fraud data is highly imbalanced.
    model = xgb.XGBClassifier(
        n_estimators=500,          # Maximum number of trees
        learning_rate=0.05,        # Step size for each tree
        max_depth=6,               # How complex each tree can get
        eval_metric='auc',
        early_stopping_rounds=50,  # Stop if validation score doesn't improve for 50 rounds
        random_state=42,
        n_jobs=-1                  # Use all available CPU cores to train faster
    )
    
    print("Training model (this will take a minute or two)...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50  # Print progress every 50 trees
    )
    
    return model

def evaluate_and_save(model, X_val, y_val):
    """Scores the model and saves it as a deployable artifact."""
    print("\nEvaluating model performance...")
    
    # Predict the *probability* of fraud, not just 0 or 1
    predictions = model.predict_proba(X_val)[:, 1]
    
    # ROC AUC is the industry standard for fraud detection
    auc_score = roc_auc_score(y_val, predictions)
    print(f"Validation ROC AUC Score: {auc_score:.4f}")
    
    # Create a 'models' directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the model
    model_path = os.path.join('models', 'xgb_fraud_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model artifact saved successfully to {model_path}")
    
    return auc_score