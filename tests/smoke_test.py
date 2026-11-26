
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

def main():
    project_dir = Path(__file__).resolve().parents[1]

    model_path = project_dir / "models" / "G09_model.joblib"
    data_path = project_dir / "data" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # Load the saved model
    model = joblib.load(model_path)
    print("Model loaded successfully.")

    # Load and prepare the dataset
    df = pd.read_csv(data_path)

    y = df["Attrition"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Target column contains unexpected values.")

    X = df.drop(
        columns=[
            "Attrition",
            "EmployeeNumber",
            "EmployeeCount",
            "Over18",
            "StandardHours"
        ],
        errors="ignore"
    )

    # Recreate the stratified test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Predict three test rows
    sample = X_test.iloc[:3]
    predictions = model.predict(sample)

    if len(predictions) != 3:
        raise ValueError("The model did not return three predictions.")

    print("\nSmoke test predictions:")
    for i, prediction in enumerate(predictions, start=1):
        print(f"Test row {i}: {prediction}")

    print("\nSMOKE TEST PASSED.")

if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"SMOKE TEST FAILED: {type(error).__name__}: {error}",
            file=sys.stderr
        )
        sys.exit(1)
