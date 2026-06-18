# IEEE-CIS Fraud Detection Pipeline (V2 Architecture)

## Overview
An end-to-end Machine Learning pipeline built to detect fraudulent credit card transactions. This project processes half a million records and trains a Gradient Boosted algorithm (XGBoost) to identify complex fraud patterns. 

The V2 architecture handles heavy mathematical features (Vesta engineered columns) alongside categorical digital fingerprinting, utilizing strict dataset alignment to prevent silent data corruption during inference.

## Performance Metric
* **Kaggle ROC AUC Score:** `~0.90`
* **Local Validation AUC:** `0.90+`

## Architecture & Version 2 Updates
The pipeline is fully modularized for production scalability:
1. **`main.py`**: The master orchestration script. Updated in V2 to include dynamic feature selection (`C`, `V`, `M`, and `id_` columns) and a robust Categorical Alignment loop to ensure the Test set perfectly mirrors the Train set's memory structure.
2. **`feature_engineering.py`**: Dynamically filters noise and handles missing values. V2 splits imputation logic: fills missing mathematical columns with `-999` (for tree-based isolation) and missing text with `'missing_category'`.
3. **`model_training.py`**: Handles train/test splitting with stratification and trains an XGBoost classifier optimized for extreme class imbalance.
4. **`inference.py`**: Processes unseen test data. V2 includes dynamic column inspection to guarantee exact feature order matching, preventing `predict_proba` flatlining.
5. **`tuning.py`**: Integrated Optuna hyperparameter optimization to mathematically maximize AUC.

## Key Insight: The Digital Fingerprint + Math
While V1 proved that digital identity features (`id_29`, `id_36`) are strong baseline predictors of fraud, V2 pushed the accuracy to the 90th percentile by injecting transaction counts and Vesta mathematical features, proving that modern fraud detection requires both device footprinting and transactional math.

## Quick Start
```bash
# Ensure RUN_TUNING is set to False in main.py for a standard training run
python main.py
