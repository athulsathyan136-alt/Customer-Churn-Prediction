from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (PROJECT_ROOT / "data" / "raw" / "Telco-Customer-Churn.csv")

def main():
    data = pd.read_csv(DATA_PATH)

    print("===== CUSTOMER CHURN DATASET =====")

    print("\nDataset shape")
    print(data.shape)

    print('\nFirst five rows:')
    print(data.head())

    print('\nColumn information:')
    data.info()

    print('\nmissing value per column:')
    print(data.isnull().sum())

    print('\nChurn Distribution:')
    print(data["Churn"].value_counts())

if __name__ == "__main__":
    main()