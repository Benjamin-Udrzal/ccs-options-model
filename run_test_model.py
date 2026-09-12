

import os

from func_test_model import test_model


def main():
    model_folder = os.environ.get("MODEL_FOLDER") or None
    test_data = os.environ.get("TEST_DATA") or None

    result = test_model(
        model_folder=model_folder,
        test_data=test_data,
        search_root=os.getcwd(),  # repo root, as checked out by the workflow
    )

    print(f"Model used: {result['model_folder']}")
    print(f"Test data used: {result['test_data_path']}")
    print(f"Predictions saved to: {result['output_path']}")
    n_trades = int(result["predictions"]["Open_Trade"].sum())
    print(f"Suggested trades to open: {n_trades} / {len(result['predictions'])} rows")


if __name__ == "__main__":
    main()