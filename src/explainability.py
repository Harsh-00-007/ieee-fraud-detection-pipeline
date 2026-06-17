import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_feature_importance(model, feature_names):
    """Extracts internal model weights to show what drives fraud detection."""
    print("\n--- EXPLAINABILITY PHASE ---")
    print("Calculating feature importance...")
    
    # 1. Extract the mathematical importance scores from XGBoost
    importances = model.feature_importances_
    
    # 2. Map them to our human-readable feature names
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # 3. Create a professional visualization
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20), palette='rocket')
    plt.title('Top 20 Fraud Indicators (XGBoost Internal Weights)')
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    
    # 4. Save the plot to a new 'reports' folder
    os.makedirs('reports', exist_ok=True)
    plot_path = os.path.join('reports', 'feature_importance.png')
    plt.savefig(plot_path)
    
    print(f"Feature importance graph saved successfully to {plot_path}")
    return importance_df