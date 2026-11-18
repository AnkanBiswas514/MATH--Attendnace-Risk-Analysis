"""MFAD Datathon CSE1 - Section RR_CSE_B
Problem Statement 2 starter template.

Replace the body of each sub-task function with the actual solution once the
PS2 sub-task questions are known (from RR_CSE_3rd B2.pdf).

Usage:
    python3 ps2_template.py   (or: make ps2)
"""

import os
import sys
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "datasets", "DATASET_N.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_dataset(path):
    """Load the allocated dataset and print a quick profile."""
    if not os.path.exists(path):
        print(f"Dataset not found: {path}")
        print("Place your allocated Dataset_N.csv in datasets/ and update DATASET_PATH.")
        sys.exit(1)
    df = pd.read_csv(path)
    print(f"Loaded dataset: {path}")
    print(f"Shape: {df.shape}")
    print(df.dtypes)
    print(df.head())
    print("Missing values per column:\n", df.isnull().sum())
    return df


# ---------------------------------------------------------------------------
# PS2 sub-task placeholders (implement per the actual PS2 questions)
# ---------------------------------------------------------------------------
def subtask_1(df):
    """PS2 - Sub-task 1."""
    result = None
    print("=== PS2 Sub-task 1 ===")
    print("TODO: implement once PS2 questions are known.")
    return result


def subtask_2(df):
    """PS2 - Sub-task 2."""
    result = None
    print("=== PS2 Sub-task 2 ===")
    print("TODO: implement once PS2 questions are known.")
    return result


def subtask_3(df):
    """PS2 - Sub-task 3."""
    result = None
    print("=== PS2 Sub-task 3 ===")
    print("TODO: implement once PS2 questions are known.")
    return result


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_dataset(DATASET_PATH)

    r1 = subtask_1(df)
    r2 = subtask_2(df)
    r3 = subtask_3(df)

    print("PS2 solution complete.")


if __name__ == "__main__":
    main()
