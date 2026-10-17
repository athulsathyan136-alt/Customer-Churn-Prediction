from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

PROCESSED_PATH = (PROJECT_ROOT / "data" / "processed" / "customer_churn_clean.csv")

def main():

    data = pd.read_csv(RAW_PATH)

    print("===== BEFORE CLEANING =====")
    print('Dataset shape: ',data.shape)
    print('TotalCharges type: ',data["TotalCharges"].dtype)

    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"],errors='coerce',)

    print("\nMissing values after conversion:")
    print(data.isnull().sum())

    data = data.dropna(subset=["TotalCharges"]).copy()
    data = data.drop_duplicates(subset=["customerID"])

    PROCESSED_PATH.parent.mkdir(parents=True,exist_ok=True)
    data.to_csv(PROCESSED_PATH,index=False)

    print("\n===== AFTER CLEANING =====")
    print("Dataset shape:", data.shape)
    print("TotalCharges type:", data["TotalCharges"].dtype)
    print("Remaining missing values:", data.isnull().sum().sum())
    print("Saved to:", PROCESSED_PATH)


if __name__ == "__main__":
    main()
