"""
Inference service for predicting total piece travel time.

Loads the trained XGBoost model and provides predictions.

Usage as CLI:
    uv run python -m vaultech_analysis.inference --die-matrix 5052 --strike2 18.3 --oee 13.5

Usage as module (for Streamlit):
    from vaultech_analysis.inference import Predictor
    predictor = Predictor()
    result = predictor.predict(die_matrix=5052, lifetime_2nd_strike_s=18.3, oee_cycle_time_s=13.5)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import json
import pandas as pd
from xgboost import XGBRegressor


class Predictor:
    def __init__(
        self,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> None:
        package_dir = Path(__file__).resolve().parent
        project_root = package_dir.parent.parent

        self.model_path = Path(model_path) if model_path else project_root / "models" / "xgboost_bath_predictor.json"
        self.metadata_path = Path(metadata_path) if metadata_path else project_root / "models" / "model_metadata.json"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            self.metadata: dict[str, Any] = json.load(f)

        self.features: list[str] = self.metadata["features"]
        self.target: str = self.metadata["target"]
        self.metrics: dict[str, Any] = self.metadata["metrics_overall"]

        self.default_oee: float = float(
            self.metadata["fillna_strategy"]["oee_cycle_time_s"]["value"]
        )

        self.model = XGBRegressor()
        self.model.load_model(self.model_path)

    def _valid_die_matrices(self) -> set[int]:
        values = set()

        for row in self.metadata.get("metrics_per_matrix", []):
            if "die_matrix" in row:
                values.add(int(row["die_matrix"]))

        return values

    def _prepare_input_row(
        self,
        die_matrix: int,
        lifetime_2nd_strike_s: float,
        oee_cycle_time_s: float | None,
    ) -> pd.DataFrame:
        if die_matrix not in self._valid_die_matrices():
            raise ValueError(
                f"Unknown die_matrix={die_matrix}. "
                f"Expected one of {sorted(self._valid_die_matrices())}"
            )

        if oee_cycle_time_s is None:
            oee_cycle_time_s = self.default_oee

        row = {
            "die_matrix": int(die_matrix),
            "lifetime_2nd_strike_s": float(lifetime_2nd_strike_s),
            "oee_cycle_time_s": float(oee_cycle_time_s),
        }

        return pd.DataFrame([row], columns=self.features)

    def predict(
        self,
        die_matrix: int,
        lifetime_2nd_strike_s: float,
        oee_cycle_time_s: float | None = None,
    ) -> dict[str, Any]:
        try:
            X = self._prepare_input_row(
                die_matrix=die_matrix,
                lifetime_2nd_strike_s=lifetime_2nd_strike_s,
                oee_cycle_time_s=oee_cycle_time_s,
            )
        except ValueError as e:
            return {
                "error": str(e),
                "die_matrix": die_matrix,
                "lifetime_2nd_strike_s": lifetime_2nd_strike_s,
                "oee_cycle_time_s": oee_cycle_time_s,
            }

        pred = float(self.model.predict(X)[0])

        used_oee = (
            float(oee_cycle_time_s)
            if oee_cycle_time_s is not None
            else self.default_oee
        )

        return {
            "predicted_bath_time_s": pred,
            "die_matrix": int(die_matrix),
            "lifetime_2nd_strike_s": float(lifetime_2nd_strike_s),
            "oee_cycle_time_s": float(used_oee),
            "model_metrics": self.metrics,
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.Series:
        required_cols = {"die_matrix", "lifetime_2nd_strike_s"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns for batch prediction: {sorted(missing)}")

        X = df.copy()

        if "oee_cycle_time_s" not in X.columns:
            X["oee_cycle_time_s"] = self.default_oee
        else:
            X["oee_cycle_time_s"] = X["oee_cycle_time_s"].fillna(self.default_oee)

        invalid_mask = ~X["die_matrix"].isin(self._valid_die_matrices())
        if invalid_mask.any():
            bad_values = sorted(X.loc[invalid_mask, "die_matrix"].dropna().unique().tolist())
            raise ValueError(
                f"Unknown die_matrix values in batch input: {bad_values}. "
                f"Expected one of {sorted(self._valid_die_matrices())}"
            )

        X = X[self.features].copy()

        preds = self.model.predict(X)
        return pd.Series(preds, index=df.index, name="predicted_bath_time_s")

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict bath travel time from early-stage features.")
    parser.add_argument("--die-matrix", type=int, required=True, help="Die matrix ID, e.g. 5052")
    parser.add_argument("--strike2", type=float, required=True, help="Lifetime at 2nd strike in seconds")
    parser.add_argument("--oee", type=float, default=None, help="Optional OEE cycle time in seconds")

    args = parser.parse_args()

    predictor = Predictor()
    result = predictor.predict(
        die_matrix=args.die_matrix,
        lifetime_2nd_strike_s=args.strike2,
        oee_cycle_time_s=args.oee,
    )

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()