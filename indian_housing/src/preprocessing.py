from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class TargetEncoder(BaseEstimator, TransformerMixin):
    """Simple target encoder that maps categories to mean target values.

    This avoids adding an external dependency and works well as a preprocessing
    step prior to modeling.
    """

    def __init__(self, cols: Optional[List[str]] = None):
        self.cols = cols or []
        self.mapping_ = {}
        self.global_mean_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.global_mean_ = float(y.mean())
        for col in self.cols:
            grouped = pd.Series(y.values, index=X[col]).groupby(level=0).mean()
            self.mapping_[col] = grouped.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.cols:
            X[col] = X[col].map(self.mapping_.get(col, {})).fillna(self.global_mean_)
        return X


class AmenitiesExtractor(BaseEstimator, TransformerMixin):
    """Transform an `amenities` text column into multi-hot binary features.

    Expects amenities in a single string column separated by commas.
    """

    def __init__(self, col: str = "amenities", top_k: int = 20):
        self.col = col
        self.top_k = top_k
        self.top_amenities_: List[str] = []

    def _split_amenities(self, s: str) -> List[str]:
        if pd.isna(s):
            return []
        parts = [p.strip().lower() for p in str(s).split(",") if p.strip()]
        return parts

    def fit(self, X: pd.DataFrame, y=None):
        all_amenities = []
        for val in X[self.col].astype(str).fillna(""):
            all_amenities.extend(self._split_amenities(val))
        counts = pd.Series(all_amenities).value_counts()
        self.top_amenities_ = counts.head(self.top_k).index.tolist()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=X.index)
        for amen in self.top_amenities_:
            out[f"amenity__{amen}"] = X[self.col].astype(str).apply(
                lambda s: 1 if amen in [p.strip().lower() for p in str(s).split(",") if p.strip()] else 0
            ).astype(int)
        return out


class DataPreprocessor(BaseEstimator, TransformerMixin):
    """Composite preprocessor for the housing dataset.

    Performs:
      - basic cleaning (fill numeric medians, categorical missing)
      - amenity multi-hot extraction
      - target encoding for `locality` & `city`
      - one-hot encoding for property/furnished
      - scaling for numeric features
    """

    def __init__(self, target_col: str = "price_in_lakhs", top_k_amenities: int = 20):
        self.target_col = target_col
        self.top_k_amenities = top_k_amenities
        self.amen_extractor: Optional[AmenitiesExtractor] = None
        self.target_encoder: Optional[TargetEncoder] = None
        self.ohe: Optional[OneHotEncoder] = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_names_: List[str] = []

    def _standardize_cols(self, X: pd.DataFrame) -> pd.DataFrame:
        # prefer lowercase column names for consistency
        X = X.copy()
        X.columns = [c.lower() for c in X.columns]
        return X

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        X = self._standardize_cols(X)

        # basic imputation values
        self.numeric_cols_ = X.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_cols_ = [c for c in X.columns if c not in self.numeric_cols_]

        # fillers
        self.numeric_fill_ = X[self.numeric_cols_].median()
        self.categorical_fill_ = {c: "missing" for c in self.categorical_cols_}

        # Amenities
        if "amenities" in X.columns:
            self.amen_extractor = AmenitiesExtractor(col="amenities", top_k=self.top_k_amenities)
            self.amen_extractor.fit(X)

        # Target encoding for locality and city
        te_cols = [c for c in ["locality", "city"] if c in X.columns]
        if te_cols and y is not None:
            self.target_encoder = TargetEncoder(cols=te_cols)
            self.target_encoder.fit(X, y)

        # One-hot for property_type and furnished_status
        ohe_cols = [c for c in ["property_type", "furnished_status"] if c in X.columns]
        if ohe_cols:
            # Create OneHotEncoder in a way that's compatible across scikit-learn versions
            try:
                self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            except TypeError:
                self.ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
            self.ohe.fit(X[ohe_cols].fillna("missing"))
            self.ohe_cols_ = ohe_cols
        else:
            self.ohe_cols_ = []

        # scaler for numeric cols
        self.scaler = StandardScaler()
        # prepare numeric matrix after fill
        numeric_matrix = X[self.numeric_cols_].fillna(self.numeric_fill_)
        self.scaler.fit(numeric_matrix)

        # build final feature name list (order matters)
        features: List[str] = []
        features.extend(self.numeric_cols_)
        if self.amen_extractor:
            features.extend([f"amenity__{a}" for a in self.amen_extractor.top_amenities_])
        if self.target_encoder:
            features.extend(te_cols)
        if self.ohe is not None:
            ohe_names = list(self.ohe.get_feature_names_out(self.ohe_cols_))
            features.extend(ohe_names)

        self.feature_names_ = features
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X = self._standardize_cols(X)

        # numeric
        numeric = X[self.numeric_cols_].fillna(self.numeric_fill_).to_numpy()

        parts = [numeric]

        # amenities
        if self.amen_extractor:
            amen_df = self.amen_extractor.transform(X)
            parts.append(amen_df.to_numpy())

        # target encoded cats
        if self.target_encoder:
            te_df = self.target_encoder.transform(X[[c for c in ["locality", "city"] if c in X.columns]])
            parts.append(te_df.to_numpy())

        # one-hot
        if self.ohe is not None and len(self.ohe_cols_)>0:
            ohe_mat = self.ohe.transform(X[self.ohe_cols_].fillna("missing"))
            # ensure dense
            try:
                from scipy import sparse as _s

                if _s.issparse(ohe_mat):
                    ohe_mat = ohe_mat.toarray()
            except Exception:
                # scipy may be unavailable or transform already dense
                pass
            parts.append(ohe_mat)

        # concatenate
        if parts:
            out = np.hstack(parts)
        else:
            out = np.empty((len(X), 0))

        # scale numeric part (first len(numeric_cols) columns)
        if out.shape[1] > 0 and len(self.numeric_cols_)>0:
            out[:, : len(self.numeric_cols_)] = self.scaler.transform(out[:, : len(self.numeric_cols_)])

        return out

    def get_feature_names_out(self) -> List[str]:
        return self.feature_names_


def build_and_save_preprocessor(X_train: pd.DataFrame, y_train: pd.Series, dest: str = "models/preprocessor.joblib", top_k_amenities: int = 20) -> Path:
    """Convenience helper to build, fit and save the preprocessing pipeline."""
    pre = DataPreprocessor(target_col="price_in_lakhs", top_k_amenities=top_k_amenities)
    pre.fit(X_train, y_train)

    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pre, dest_path)
    return dest_path


__all__ = ["DataPreprocessor", "build_and_save_preprocessor", "TargetEncoder", "AmenitiesExtractor"]
