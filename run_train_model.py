"""
Entry point for the "Train CCS Model" GitHub Actions workflow.

By default, trains on whatever the LATEST committed "<date> train Model Run" folder is,
anywhere in the repo checkout. Saves model artifacts inside that same folder, so a model
is always traceable to exactly the data it was trained on — the workflow then commits
that folder back to GitHub, the same way run_daily_import.py does for imports.

To train on a specific dated run instead of the latest one, set the TRAIN_IMPORT_FOLDER
environment variable (e.g. via workflow_dispatch inputs) to either the folder's bare name
(e.g. "2026-08-13__13.37 train Model Run") or a full path.
"""

import os
import sys

from func_train import func_train_model


def main():
    train_import_folder = os.environ.get("TRAIN_IMPORT_FOLDER") or None

    result = func_train_model(
        train_import_folder=train_import_folder,
        search_root=os.getcwd(),  # repo root, as checked out by the workflow
    )

    print(f"Trained from: {result['found_folder']}")
    print(f"Artifacts saved to: {result['output_dir']}")
    if result["reg_metrics"]:
        print(f"Test regression MAE: {result['reg_metrics']['MAE_single_model']:.4f}")
    if result["cls_metrics"]:
        print(f"Test classification accuracy: {result['cls_metrics']['Accuracy']:.4f}")


if __name__ == "__main__":
    main()