# FraudShield — Fraud Detection Dashboard

An interactive machine-learning application for identifying potentially fraudulent transactions, reviewing suspicious activity, and maintaining investigation records.

## 📊 Dashboard Preview

![Uploading image.png…]()


## 🚀 Live Demo

**Try FraudShield online:** [Launch FraudShield](https://fraudshield-tfhpmfhblhtzoxlvgmum9a.streamlit.app/)

Explore the interactive fraud detection dashboard, analyze sample transactions, view fraud probability charts, and review transaction history.

## Overview

FraudShield uses an XGBoost classification model to estimate transaction fraud risk. The Streamlit dashboard lets users explore predictions, review risk indicators, visualize results, and maintain investigation notes.

## Features

- **Fraud prediction:** Classifies transactions using a trained XGBoost model.
- **Transaction analysis:** Supports sample transactions and manual entry.
- **Risk visualization:** Displays prediction distributions, fraud-versus-legitimate summaries, and fraud probability trends.
- **Persistent history:** Stores transaction history in SQLite.
- **Investigation workflow:** Records review statuses and analyst notes.
- **CSV exports:** Exports transaction history and flagged transactions.
- **History management:** Provides a way to clear saved transaction history.

## Technology Stack

- Python
- Streamlit
- XGBoost
- scikit-learn
- pandas and NumPy
- Plotly
- SQLite

## Getting Started

### Prerequisites

- Python installed on your computer
- Git, if you want to clone the repository
- The project files and a compatible Python environment

### Installation

1. Clone or download this repository.
2. Open a terminal in the project directory.
3. Create and activate a virtual environment if you do not already have one.
4. Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

5. Start the application:

```bash
python -m streamlit run app.py
```

6. Open the local URL displayed in the terminal, usually `http://localhost:8501`.

**Note:** Reproducing the complete application may also require the trained model files and any required data files. Confirm their availability and compatibility before running the project in a fresh environment.

## Project Structure


fraudshield/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── fraudshield_xgb_model.json

**Local development files (not included in the repository):**
- `fraudshield_history.db` — local transaction history
- `fraudshield_sample.csv` — local sample dataset
- `fraudshield_model.joblib` — local model artifact, if used

The files listed above reflect the current development workspace. Review the files and adjust this structure before publishing the repository.

## Model Evaluation

Evaluate the model on a suitable held-out dataset before reporting performance. Useful metrics include:

- Precision
- Recall
- F1-score
- Confusion matrix
- ROC-AUC, where appropriate

Fraud detection often involves imbalanced datasets, so accuracy alone may not adequately measure performance.

## Privacy and Security

- Do not commit API keys, passwords, private data, or other secrets.
- Use synthetic or appropriately anonymized transaction data for demonstrations.
- Review database contents and model artifacts before making the repository public.

## Limitations

FraudShield provides model-based risk estimates, not definitive proof of fraud. Predictions should be reviewed alongside appropriate evidence and investigation procedures.

## Future Improvements

- Model evaluation and threshold tuning
- Improved data validation and error handling
- Authentication and role-based access
- Deployment with secure configuration
- Model monitoring and drift detection

## Disclaimer

This project is intended for educational and demonstration purposes. It should not be used as the sole basis for financial or fraud-related decisions.

## Author

Name - Tushar Singh , Github Profile - https://github.com/Tushar7652  and LinkedIn profile - https://www.linkedin.com/in/tushar-singh-7885bb226/?isSelfProfile=true

## Dataset Attribution

This project was developed using a credit-card fraud detection dataset with the schema `Time`, `V1`–`V28`, `Amount`, and `Class`.

- **Dataset:** [ULB Credit Card Fraud Detection dataset on Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Reported dataset licence:** Database Contents License (DbCL-1.0).
- **Redistribution:** The dataset CSV is not included in this repository. The trained model is distributed only subject to the applicable licence terms and conditions.

Users should review the dataset's current licence and applicable Open Data Commons terms before redistributing the dataset, trained model, or derivatives.
