from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT/"models"/"best_churn_model.joblib"
)

def main():

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model Not found:{MODEL_PATH}")

    model_package = joblib.load(MODEL_PATH)

    model = model_package["model"]
    feature_names = model_package["feature_names"]

    customers = {
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 75.0,
        "TotalCharges": 900.0,
    }

    customer_data = pd.DataFrame(0,index=[0],columns=feature_names,)

    for feature, value in customers.items():
        if feature in customer_data.columns:
            customer_data.loc[0,feature] = value

    churn_probability = model.predict_proba(customer_data)[0,1]

    prediction = model.predict(customer_data)[0]

    print("===== CUSTOMER CHURN PREDICTION =====")
    print("Customer tenure:", customers["tenure"], "months")
    print(
        "Monthly charges: $",
        customers["MonthlyCharges"],
    )
    print(
        "Predicted churn probability:",
        f"{churn_probability:.2%}",
    )

    if prediction == 1:
        print("Prediction: Customer is at risk of churn")
    else:
        print("Prediction: Customer is predicted to stay")


if __name__ == "__main__":
    main()
