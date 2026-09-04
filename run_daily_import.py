"""
Entry point for the GitHub Actions workflow.

Reads the Polygon API key from an environment variable (set from a GitHub
Actions secret — never hard-code it here), runs func_import for the daily
("dtest") case, and lets func_import save its own CSVs into ./outputs so the
workflow can commit them back into the repo.
"""

import os
import sys

from func_import import func_import


def main():
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("ERROR: POLYGON_API_KEY environment variable is not set.")
        sys.exit(1)

    # "outputs" is relative to the repo root (GitHub Actions checks the repo
    # out into the working directory the job runs in), so files land at
    # <repo>/outputs/... and can be committed straight back.
    output_dir = os.path.join(os.getcwd(), "dtest_outputs")

    result = func_import(
        polygon_api_key=api_key,
        run_reason="dtest",
        default_settings=True,
        output_dir=output_dir,
    )

    print(f"Saved files to: {result['output_dir']}")
    print(f"Testing dataset rows: {len(result['testing_dataset'])}")


if __name__ == "__main__":
    main()
