from pathlib import Path
from typing import Optional, Union

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = "data/raw/india_housing_prices.csv"
REQUIRED_COLUMNS = [
    "State",
    "City",
    "Locality",
    "Property_Type",
    "BHK",
    "Size_in_SqFt",
    "Price_in_Lakhs",
]


def resolve_data_path(file_path: Optional[Union[str, Path]] = None) -> Path:
    """Resolve a dataset path from common project-relative locations."""
    candidate = file_path or DEFAULT_DATA_PATH
    path = Path(candidate)

    if path.is_absolute() and path.exists():
        return path

    search_roots = [Path.cwd(), PROJECT_ROOT, PROJECT_ROOT.parent]
    for root in search_roots:
        full_path = root / path
        if full_path.exists():
            return full_path

    return PROJECT_ROOT / path


def load_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load and validate the raw housing dataset."""
    resolved_path = resolve_data_path(file_path)

    if not resolved_path.exists():
        raise FileNotFoundError(f"Dataset not found. Tried: {resolved_path}")

    df = pd.read_csv(resolved_path)

    if df.empty:
        raise ValueError("Dataset is empty.")

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    print("Data ingestion successful!")
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    return df


def ingest_data(file_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Compatibility wrapper for the ingestion function."""
    return load_data(file_path)


def main() -> None:
    df = load_data()
    print(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    print(df.head())


if __name__ == "__main__":
    main()

