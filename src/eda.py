from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT/"data"/"processed"/"customer_churn_clean.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"

def main():
    data = pd.read_csv(DATA_PATH)

    REPORTS_DIR.mkdir(parents=True,exist_ok=True)

    print("======CUSTOMER CHURN EDA========")

    print('\nDataset shape: ')
    print(data.shape)

    print('\nChurn counts: ')
    print(data["Churn"].value_counts())

    print('\nChurn Percentage: ')
    print((data["Churn"].value_counts(normalize=True)*100).round(2))

    print("\nChurn rate by contract type:")
    contract_churn = (
        data.groupby("Contract",observed=True)["Churn"].apply(lambda values : (values == "Yes").mean()*100).round(2)
    )

    print(contract_churn)
    print("\nAverage tenure by churn status:")
    print(data.groupby("Churn",observed=True)["tenure"].mean().round(2))

    print("\nAverage monthly charges by churn status:")
    print(data.groupby("Churn",observed=True)["MonthlyCharges"].mean().round(2))

    churn_count = data["Churn"].value_counts()

    plt.figure(figsize=(7,5))
    churn_count.plot(kind='bar')
    plt.title("Customer churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "churn_distribution.png")
    plt.close()

    contract_churn.sort_values().plot(kind="bar",figsize=(8,5))
    plt.title("Churn Rate by Contract Type")
    plt.xlabel("Contract Type")
    plt.ylabel("Churn Rate (%)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "churn_by_contract.png")
    plt.close()

    data.boxplot(column="tenure",by="Churn",figsize=(7,5))
    plt.title("Customer Tenure by Churn Status")
    plt.suptitle("")
    plt.xlabel("Churn")
    plt.ylabel("Tenure (months)")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "tenure_by_churn.png")
    plt.close()
    print("\nCharts saved in:", REPORTS_DIR)
    print("EDA completed successfully.")

if __name__ == "__main__":
    main()
