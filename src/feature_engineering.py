
from pathlib import Path

import pandas as pd


# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_clean.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "customer_churn_features.csv"
)


def main():
    # 2. Load the cleaned dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    print("===== BEFORE FEATURE ENGINEERING =====")
    print("Dataset shape:", data.shape)

    # 3. Remove the customer ID because it is not a useful
    # predictive feature
    if "customerID" in data.columns:
        data = data.drop(columns=["customerID"])

    # 4. Convert the target column into numbers
    # No = 0, Yes = 1
    data["Churn"] = data["Churn"].map(
        {"No": 0, "Yes": 1}
    )

    # Check for unexpected target values
    if data["Churn"].isnull().any():
        raise ValueError(
            "Unexpected values found in the Churn column."
        )

    # 5. Separate features from the target
    target = data["Churn"]
    features = data.drop(columns=["Churn"])

    # 6. Find categorical columns
    categorical_columns = features.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()

    print("\nCategorical columns:", len(categorical_columns))

    # 7. Convert categorical values into numeric columns
    # Example: Contract becomes Contract_Month-to-month, etc.
    features = pd.get_dummies(
        features,
        columns=categorical_columns,
        dtype=int,
    )

    # 8. Combine features and target
    prepared_data = features.copy()
    prepared_data["Churn"] = target.astype(int)

    # 9. Verify the prepared dataset
    print("\n===== AFTER FEATURE ENGINEERING =====")
    print("Prepared dataset shape:", prepared_data.shape)
    print(
        "Remaining missing values:",
        prepared_data.isnull().sum().sum(),
    )
    print("\nTarget distribution:")
    print(prepared_data["Churn"].value_counts())

    # 10. Create the output folder and save the CSV
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nSaving to:", OUTPUT_PATH)

    prepared_data.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    # 11. Confirm that the file really exists
    if not OUTPUT_PATH.is_file():
        raise RuntimeError(
            f"Output file was not created: {OUTPUT_PATH}"
        )

    print("\n===== SAVE SUCCESSFUL =====")
    print("File exists:", OUTPUT_PATH.is_file())
    print("File size:", OUTPUT_PATH.stat().st_size, "bytes")
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
