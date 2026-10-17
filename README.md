
# Customer Churn Prediction

A machine learning project that predicts whether a telecommunications customer is likely to leave a service provider.

## Project Overview

Customer churn prediction helps businesses identify customers who may leave and take proactive steps to improve retention.

This project includes:
- Data cleaning and exploratory data analysis
- Feature engineering and categorical encoding
- Logistic Regression and Random Forest model comparison
- A reusable scikit-learn preprocessing and prediction pipeline
- A REST API built with FastAPI

## Machine Learning

**Selected model:** Scaled Logistic Regression pipeline

Evaluation on a stratified 80/20 train-test split:

| Metric | Result |
|---|---:|
| ROC-AUC | 0.8351 |
| Churn recall | 0.80 |
| Churn precision | 0.49 |
| Churn F1-score | 0.61 |
| Test records | 1,407 |

The model prioritizes identifying customers at risk of churn. Its predictions are estimates and should not be treated as guarantees.

## Dataset

IBM Telco Customer Churn dataset.

The cleaned dataset contains 7,032 customer records after removing rows with missing `TotalCharges` values.

## Project Structure

```text
Customer-Churn-Prediction/
├── api/
│   └── main.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── src/
├── requirements.txt
└── README.md
```

## Installation

Python 3.12 is recommended for this environment.

1. Clone the repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   cd Customer-Churn-Prediction
   ```

2. Create and activate a virtual environment on Windows:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

## Run the API

From the project root:

```powershell
python -m uvicorn api.main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

Health check:

http://127.0.0.1:8000/health

## API Endpoint

### POST `/predict`

Submit customer details as JSON to receive a churn probability and prediction.

The API expects the original customer feature fields, excluding `customerID` and `Churn`.

Example response:

```json
{
  "churn_probability": 0.8517,
  "prediction": "Churn risk"
}
```

This response is illustrative; actual results depend on the supplied customer data and saved model.

## Limitations

- Performance depends on the training dataset and its representativeness.
- Predictions may produce false positives and false negatives.
- Churn probability is not a guarantee of customer behavior.
- The model should be monitored and validated before real business deployment.

## Future Improvements

- Tune the classification threshold based on business costs.
- Add automated tests for the API.
- Containerize the application with Docker.
- Deploy the API to a cloud platform.
- Monitor model performance and data drift.

## Author

Athul Sathyan

GitHub: https://github.com/athulsathyan136-alt
