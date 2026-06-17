# IEEE-CIS Fraud Detection Pipeline

## Overview
An end-to-end Machine Learning pipeline built to detect fraudulent credit card transactions. This project processes half a million records and trains a Gradient Boosted algorithm (XGBoost) to identify complex fraud patterns based on digital fingerprinting and transaction metadata.

## Performance Metric
* **Validation ROC AUC:** `0.9047`

## Architecture
The pipeline is fully modularized for production scalability:
1. **`data_loader.py`**: Ingests and merges 1M+ rows of identity and transaction data.
2. **`feature_engineering.py`**: Dynamically filters noise, imputes missing values, and casts string objects to native Pandas categories.
3. **`model_training.py`**: Handles train/test splitting with stratification and trains an XGBoost classifier optimized for extreme class imbalance.
4. **`inference.py`**: Processes unseen test data and generates final probability predictions.
5. **`explainability.py`**: Extracts internal tree weights to visualize feature importance.

## Key Insight: The Digital Fingerprint
Instead of relying purely on transaction amounts, the model identified `id_29` and `id_36` (digital identity features representing network and device footprints) as the highest predictors of fraud. This proves the algorithm successfully learned to detect device emulators and IP mismatches standard in modern financial crime.

## Quick Start
```bash
python main.py