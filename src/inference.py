import pandas as pd
import os

def generate_submission(model, test_processed: pd.DataFrame):
    """Feeds the test data into the model and generates a CSV of fraud probabilities."""
    print("\n--- INFERENCE PHASE ---")
    print("Generating final predictions for the test set...")
    
    # 1. Extract IDs (We need these for the final output file)
    submission = pd.DataFrame({'TransactionID': test_processed['TransactionID']})
    
    # 2. Drop the ID column so we only feed the 50 math features into XGBoost
    X_test = test_processed.drop(['TransactionID'], axis=1)
    
    # 3. Predict the PROBABILITY of fraud (using [:, 1] to get the "Fraud" column)
    predictions = model.predict_proba(X_test)[:, 1]
    submission['isFraud'] = predictions
    
    # 4. Save the artifact
    output_path = os.path.join('data', 'submission.csv')
    submission.to_csv(output_path, index=False)
    
    print(f"Inference complete! Saved {len(submission)} predictions to {output_path}")
    return submission