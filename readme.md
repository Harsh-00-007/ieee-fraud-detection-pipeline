# IEEE-CIS Fraud Detection Pipeline (V2)

## Project Overview
An end-to-end Machine Learning microservice designed to detect fraudulent credit card transactions. This project demonstrates a production-ready pipeline, covering data ingestion, feature engineering, high-performance training, and API deployment.

## Key Technical Achievements
* **Model Performance:** Achieved a **0.90+ ROC AUC** on Kaggle by engineering 60+ mathematical (Vesta) and counting (C-columns) features.
* **MLOps/Deployment:** Developed a high-speed inference microservice using **FastAPI** and **Uvicorn**, allowing real-time transaction scoring via a RESTful API.
* **Engineering Robustness:** Solved data corruption issues caused by categorical feature mismatch between training/inference sets by implementing a strict **Categorical Alignment** architecture.
* **Optimization:** Integrated **Optuna** for hyperparameter tuning, resulting in a 5% increase in validation accuracy.

## Tech Stack
* **Language:** Python
* **ML:** XGBoost, Scikit-Learn, Pandas
* **Deployment:** FastAPI, Uvicorn, Pydantic
* **Utilities:** Optuna, Git, Joblib

## Quick Start
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the API: `uvicorn app:app --reload`
4. Access the API documentation at: `http://127.0.0.1:8000/docs`