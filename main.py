from src.data_loader import load_and_merge
import pandas as pd
from src.feature_engineering import run_feature_pipeline
from src.model_training import prepare_data, train_xgboost, evaluate_and_save
from src.inference import generate_submission
from src.explainability import plot_feature_importance


# ==========================================
# 1. FEATURE SELECTION CONFIGURATION
# ==========================================
# Define base features
MY_FEATURES = [
    'TransactionAmt', 'ProductCD', 'addr1', 'addr2', 'card1',
    'card2', 'card3', 'card4', 'card5', 'card6','P_emaildomain',
    'R_emaildomain', 'DeviceType', 'DeviceInfo'
]

MY_FEATURES += [f'C{i}' for i in range(1, 15)]     
MY_FEATURES += [f'V{i}' for i in range(1, 51)]        
MY_FEATURES += [f'M{i}' for i in range(1, 10)]        # M1 to M9
MY_FEATURES += [f'id_{i}' for i in range(12, 39)]     # id_12 to id_38


# ==========================================
# 2. PIPELINE EXECUTION
# ==========================================
if __name__ == "__main__":
    print("Initializing IEEE Fraud Detection Pipeline...\n")
    
    # Step 1: Load and Merge the raw CSVs into memory
    train, test = load_and_merge()

    # Step 2: Process the Training Set
    print("\nProcessing Training Data...")
    train_processed = run_feature_pipeline(train, features_to_keep=MY_FEATURES)
    # Step 3: Process the Testing Set identically to prevent data leakage
    print("\nProcessing Test Data...")
    test_processed = run_feature_pipeline(test, features_to_keep=MY_FEATURES)

    # ---> PASTE IT EXACTLY HERE <---
    print("Aligning categorical data structures...")
    for col in train_processed.columns:
        if train_processed[col].dtype.name == 'category' and col in test_processed.columns:
            test_processed[col] = pd.Categorical(
                test_processed[col], 
                categories=train_processed[col].cat.categories
            )

    print("Status: READY FOR MODEL TRAINING")

    # Step 4: Final Output Validation
    print("\n========================================")
    print("          PIPELINE SUMMARY              ")
    print("========================================")
    print(f"Final Training Data Shape: {train_processed.shape}")
    print(f"Final Test Data Shape:     {test_processed.shape}")
    print("Status: READY FOR MODEL TRAINING")
    

    X_train, X_val, y_train, y_val = prepare_data(train_processed)
    
    trained_model = train_xgboost(X_train, y_train, X_val, y_val)
    
    # Step 5: Evaluation and Serialization
    evaluate_and_save(trained_model, X_val, y_val)
    # Step 6: Inference (Predicting the Test Set)
    generate_submission(trained_model, test_processed)
    
    # Step 7: Explainability (Visualizing the AI)
    # We drop 'TransactionID' and 'isFraud' to get the exact list of 50 features the model learned on
    feature_names = train_processed.drop(['TransactionID', 'isFraud'], axis=1).columns
    plot_feature_importance(trained_model, feature_names)
    
    print("\n[SUCCESS] IEEE Fraud Detection Pipeline execution fully completed!")