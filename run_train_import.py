"""
Entry point for the "Train Data Import" GitHub Actions workflow.

Same idea as run_daily_import.py, but calls func_import with run_reason="train" instead of
"dtest" — this is what actually produces the "<date> train Model Run" folder that
CCS_Model_Train_v7.py (train_model()) looks for. Saves into ./train_outputs so the workflow
can commit it back to the repo, same pattern as the daily import.
"""

import os
import sys

from func_import import func_import


def main():
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("ERROR: POLYGON_API_KEY environment variable is not set.")
        sys.exit(1)

    # "train_outputs" is relative to the repo root, so files land at
    # <repo>/train_outputs/... and can be committed straight back.
    output_dir = os.path.join(os.getcwd(), "train_outputs")

    result = func_import(
        polygon_api_key=api_key,
        run_reason="train",
        default_settings=True,
        output_dir=output_dir,
    )

    print(f"Saved files to: {result['output_dir']}")
    if result["training_dataset"] is not None:
        print(f"Training dataset rows: {len(result['training_dataset'])}")
    print(f"Testing dataset rows: {len(result['testing_dataset'])}")


if __name__ == "__main__":
    main()