"""Executable B2B/B2C solution for the GSM smartphone model.

This script is the practical interface built from the modeling outputs.
It does not retrain the model. Instead, it loads saved artifacts from modeling/models and modeling/outputs so a user can:

1. ask for a business fair-price guide for a planned smartphone, or
2. enter a budget and receive ranked smartphone recommendations.

The comments explain why each inference step is needed, especially because the training data was encoded and scaled during preprocessing.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


class TwoWaySmartphoneSolution:
    """Load trained artifacts and provide business/user-facing predictions.

    The class wraps the final outputs into one reusable object. 
    This keeps the demo simple: 
    the user does not need to know where every CSV/model file is saved, and the modeling workflow exposes one clear two-way solution interface.
    """

    def __init__(self, root_dir: Path | None = None) -> None:
        # Resolve paths from the repository root so the script works after the project is cloned or uploaded to Colab. 
        # No absolute local path is required.
        self.root_dir = Path(root_dir) if root_dir else Path(__file__).resolve().parents[1]
        self.modeling_dir = self.root_dir / "modeling"
        self.output_dir = self.modeling_dir / "outputs"
        self.model_dir = self.modeling_dir / "models"

        # The trained price model was saved as a package, not as a bare model, because inference also needs target mode, feature names, and feature
        # medians to build a valid input row.
        self.package = joblib.load(self.model_dir / "best_price_regressor.pkl")
        self.model = self.package["model"]
        self.target_mode = self.package["target_mode"]
        self.feature_names = list(self.package["feature_names"])
        self.feature_medians = dict(self.package["feature_medians"])
        # Optional tables make the output more explainable.
        # The class can still load if a non-critical table is missing, but prediction requires the saved model package.
        self.numeric_reference = self._load_optional_csv(self.model_dir / "numeric_feature_reference.csv")
        self.brand_premium = self._load_optional_csv(self.output_dir / "brand_premium_summary.csv")
        self.adjusted_brand_premium = self._load_optional_csv(self.output_dir / "adjusted_brand_premium_summary.csv")
        self.price_drivers = self._load_optional_csv(self.output_dir / "two_way_price_driver_summary.csv")
        self.segment_df = self._load_optional_csv(self.output_dir / "df_with_segments.csv")
        self.outliers = self._load_optional_csv(self.output_dir / "value_outliers.csv")
        self.outlier_keys = self._make_outlier_keys()

    @staticmethod
    def _load_optional_csv(path: Path) -> pd.DataFrame:
        """Load a support CSV when it exists; otherwise return an empty table.

        This makes the demo robust. For example, recommendation can still fail
        clearly if segment outputs are missing, while business prediction can
        still run even if some explanation summary files are unavailable.
        """
        if path.exists():
            return pd.read_csv(path)
        return pd.DataFrame()

    def _make_outlier_keys(self) -> set[str]:
        """Create stable phone keys for value-outlier lookup.

        The outlier CSV and segment CSV are separate outputs.
        Combining OEM and model name lets the recommender mark whether each candidate is also a detected value-for-money outlier.
        """
        if self.outliers.empty or not {"oem", "model"}.issubset(self.outliers.columns):
            return set()
        return set((self.outliers["oem"].astype(str) + "||" + self.outliers["model"].astype(str)).tolist())

    def _empty_feature_row(self) -> pd.DataFrame:
        """Start a model input row from training medians.

        Users may enter only a few planned-phone specs. 
        Median filling creates a reasonable neutral baseline for unspecified features and keeps the model input shape identical to training.
        """
        row = {feature: float(self.feature_medians.get(feature, 0.0)) for feature in self.feature_names}
        return pd.DataFrame([row], columns=self.feature_names)

    def _standardize_numeric(self, raw_feature: str, value: float | None) -> tuple[str, float] | None:
        """Convert a raw user-entered numeric spec into the scaled model feature.

        The regression model was trained on StandardScaler-transformed features.
        Without this conversion, raw values such as 5000 mAh battery capacity would be on the wrong scale and predictions would be invalid.
        """
        if value is None or self.numeric_reference.empty:
            return None
        match = self.numeric_reference[self.numeric_reference["raw_feature"].eq(raw_feature)]
        if match.empty:
            return None
        ref = match.iloc[0]
        std = float(ref["std"]) if float(ref["std"]) != 0 else 1.0
        return str(ref["model_feature"]), (float(value) - float(ref["mean"])) / std

    def _set_brand(self, X: pd.DataFrame, brand: str | None) -> None:
        """Set the one-hot brand columns for a business price query.

        Brand is one of the project objectives, so the demo must let users test how a planned device's brand group changes the predicted fair price.
        Unknown brands fall back to "Other" when that column exists.
        """
        brand_cols = [c for c in self.feature_names if c.startswith("cat__brand_group_")]
        for col in brand_cols:
            X.loc[0, col] = 0.0
        if not brand:
            return
        exact = f"cat__brand_group_{brand}"
        if exact in X.columns:
            X.loc[0, exact] = 1.0
        elif "cat__brand_group_Other" in X.columns:
            X.loc[0, "cat__brand_group_Other"] = 1.0

    def _infer_premium_brand(self, brand: str | None) -> float | None:
        """Infer the preprocessing premium-brand flag from brand summary output.

        This keeps the business demo aligned with preprocessing.
        If the user only enters a brand name, the model can still infer whether that brand was treated as premium in the training data.
        """
        if not brand or self.brand_premium.empty or "avg_is_premium_brand" not in self.brand_premium.columns:
            return None
        row = self.brand_premium[self.brand_premium["brand_group"].eq(brand)]
        if row.empty:
            return None
        return float(row.iloc[0]["avg_is_premium_brand"])

    def business_price_guide(self, specs: dict[str, Any]) -> dict[str, Any]:
        """Predict an objective EUR price guideline for a planned phone.

        This is the B2B side of the project.
        It receives planned hardware/brand specs, converts them into the encoded/scaled model space, predicts a fair EUR price, 
        and compares that prediction with the planned price when one is provided.
        """
        X = self._empty_feature_row()
        brand = specs.get("brand") or specs.get("brand_group")
        self._set_brand(X, brand)

        # Derive helper features when the user provides enough raw information.
        # This mirrors preprocessing feature engineering, but only for the fields that are practical to enter in a command-line demo.
        launch_year = specs.get("launch_year")
        phone_age = specs.get("phone_age")
        if phone_age is None and launch_year is not None:
            phone_age = max(0, datetime.now().year - float(launch_year))
        network_generation = specs.get("network_generation")
        has_5g = specs.get("has_5g")
        if has_5g is None and network_generation is not None:
            has_5g = 1.0 if float(network_generation) >= 5 else 0.0
        has_4g_or_more = specs.get("has_4g_or_more")
        if has_4g_or_more is None and network_generation is not None:
            has_4g_or_more = 1.0 if float(network_generation) >= 4 else 0.0
        is_premium_brand = specs.get("is_premium_brand")
        if is_premium_brand is None:
            is_premium_brand = self._infer_premium_brand(brand)
        resolution_width = specs.get("resolution_width_px")
        resolution_height = specs.get("resolution_height_px")
        resolution_total = specs.get("resolution_total_px")
        if resolution_total is None and resolution_width is not None and resolution_height is not None:
            resolution_total = float(resolution_width) * float(resolution_height)
        aspect_ratio = specs.get("aspect_ratio")
        if aspect_ratio is None and resolution_width not in (None, 0) and resolution_height is not None:
            aspect_ratio = float(resolution_height) / float(resolution_width)

        # Map raw business-input names to the numeric features used by the model.
        # Each value is standardized before it is injected into the model row.
        raw_to_cli = {
            "is_premium_brand": is_premium_brand,
            "launch_year": launch_year,
            "launch_month": specs.get("launch_month"),
            "phone_age": phone_age,
            "body_weight_g": specs.get("body_weight_g"),
            "display_size_in": specs.get("display_size_in"),
            "screen_to_body_pct": specs.get("screen_to_body_pct"),
            "resolution_width_px": resolution_width,
            "resolution_height_px": resolution_height,
            "resolution_total_px": resolution_total,
            "aspect_ratio": aspect_ratio,
            "ppi": specs.get("ppi"),
            "ram_gb": specs.get("ram_gb"),
            "storage_gb": specs.get("storage_gb"),
            "battery_capacity_mah": specs.get("battery_capacity_mah"),
            "fast_charging_w": specs.get("fast_charging_w"),
            "main_camera_max_mp": specs.get("main_camera_max_mp"),
            "selfie_camera_max_mp": specs.get("selfie_camera_max_mp"),
            "network_generation": network_generation,
            "has_5g": has_5g,
            "has_4g_or_more": has_4g_or_more,
            "has_nfc": specs.get("has_nfc"),
            "sensor_count": specs.get("sensor_count"),
            "has_fingerprint": specs.get("has_fingerprint"),
            "has_gyro": specs.get("has_gyro"),
            "has_compass": specs.get("has_compass"),
        }
        if specs.get("spec_score_0_100") is not None:
            # The trained regression input intentionally excludes spec_score_0_100 to avoid target leakage, so this value is documented but not injected.
            pass
        for raw_feature, value in raw_to_cli.items():
            pair = self._standardize_numeric(raw_feature, value)
            if pair and pair[0] in X.columns:
                X.loc[0, pair[0]] = pair[1]

        # The saved model may have been trained on log(price).
        # Convert back to EUR so the result is directly interpretable by companies.
        pred = float(self.model.predict(X)[0])
        if self.target_mode == "log_price_eur":
            pred = float(np.expm1(pred))
        pred = max(pred, 1.0)

        budget = specs.get("budget_price_eur")
        gap = None if budget is None else float(budget) - pred
        # Convert the numeric price gap into a short guidance label so the output can be used directly as a business-side pricing decision aid.
        if gap is None:
            guidance = "no_planned_price_to_compare"
        elif abs(gap) <= max(30, pred * 0.08):
            guidance = "planned_price_is_near_model_guideline"
        elif gap > 0:
            guidance = "planned_price_is_above_model_guideline"
        else:
            guidance = "planned_price_is_below_model_guideline"

        return {
            "brand": brand,
            "predicted_price_eur": pred,
            "budget_price_eur": budget,
            "price_gap_eur": gap,
            "guidance": guidance,
            "top_price_factors": self._top_price_factors(),
            "brand_premium_note": self._brand_note(brand),
        }

    def _top_price_factors(self) -> str:
        """Return the strongest price drivers for explainable output."""
        if self.price_drivers.empty or "feature" not in self.price_drivers.columns:
            return "price driver summary not available"
        return ", ".join(self.price_drivers["feature"].head(5).astype(str).tolist())

    def _brand_note(self, brand: str | None) -> str:
        """Return a compact brand-premium note for the selected brand."""
        if not brand:
            return "no brand specified"
        parts = []
        if not self.brand_premium.empty:
            row = self.brand_premium[self.brand_premium["brand_group"].eq(brand)]
            if not row.empty:
                r = row.iloc[0]
                parts.append(f"avg_price={r['avg_price_eur']:.1f} EUR, count={int(r['count'])}")
        if not self.adjusted_brand_premium.empty:
            row = self.adjusted_brand_premium[self.adjusted_brand_premium["brand_group"].eq(brand)]
            if not row.empty:
                r = row.iloc[0]
                parts.append(f"adjusted_premium={r['avg_adjusted_brand_premium_eur']:.1f} EUR")
        return "; ".join(parts) if parts else "brand premium row unavailable"

    def recommend_for_user(
        self,
        budget_eur: float,
        top_n: int = 10,
        segment: str | None = None,
        brand: str | None = None,
    ) -> pd.DataFrame:
        """Recommend phones that maximize Performance-to-Price Ratio under a budget.

        This is the B2C side of the project.
        The budget constraint is applied before ranking so every returned phone is affordable for the user.
        """
        if self.segment_df.empty:
            raise FileNotFoundError("outputs/df_with_segments.csv가 필요합니다.")
        df = self.segment_df.copy()
        if "performance_to_price_ratio" not in df.columns:
            # Recompute PPR if an older segment output does not already contain
            # it. This keeps the demo backward-compatible with saved outputs.
            df["performance_to_price_ratio"] = df["spec_score_0_100"] / df["price_eur"]
        key = df["oem"].astype(str) + "||" + df["model"].astype(str)
        df["is_value_outlier"] = key.isin(self.outlier_keys)
        candidates = df[df["price_eur"] <= float(budget_eur)].copy()
        if segment:
            candidates = candidates[candidates["segment"].eq(segment)].copy()
        if brand:
            candidates = candidates[candidates["brand_group"].eq(brand)].copy()
        if candidates.empty:
            return candidates
        candidates["budget_utilization"] = candidates["price_eur"] / float(budget_eur)
        candidates["recommendation_reason"] = candidates.apply(self._recommendation_reason, axis=1)
        optional_cols = [
            col for col in ["market_price_discount_ratio", "expected_market_price_eur", "market_underpriced_outlier"]
            if col in candidates.columns
        ]
        cols = [
            "recommendation_rank", "oem", "model", "brand_group", "price_eur", "budget_utilization",
            "price_tier", "segment", "spec_score_0_100", "value_score",
            "performance_to_price_ratio", *optional_cols, "is_value_outlier", "recommendation_reason",
        ]
        # Ranking priority follows the project definition of value:
        # maximize PPR first, then prefer stronger value score, confirmed value outliers, stronger specs, and finally lower price as a tie-breaker.
        ranked = (
            candidates.sort_values(
                ["performance_to_price_ratio", "value_score", "is_value_outlier", "spec_score_0_100", "price_eur"],
                ascending=[False, False, False, False, True],
            )
            .head(top_n)
            .reset_index(drop=True)
        )
        ranked.insert(0, "recommendation_rank", range(1, len(ranked) + 1))
        return ranked[cols]

    @staticmethod
    def _recommendation_reason(row: pd.Series) -> str:
        """Build a human-readable reason for each recommendation row."""
        discount = row.get("market_price_discount_ratio", np.nan)
        discount_note = "" if pd.isna(discount) else f", market_discount={discount:.1%}"
        return (
            f"PPR={row['performance_to_price_ratio']:.3f}, value={row['value_score']:.2f}, "
            f"spec={row['spec_score_0_100']:.1f}{discount_note}, "
            f"RAM={row['ram_gb']:.0f}GB, camera={row.get('main_camera_max_mp', np.nan):.0f}MP"
        )

    def run_demo(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate demo business and user outputs.

        Static examples make it easy to verify the two-way solution without manually typing long JSON input during a quick modeling check.
        """
        business_specs = [
            {
                "scenario": "B2B_midrange_launch",
                "brand": "Samsung",
                "ram_gb": 8,
                "storage_gb": 128,
                "battery_capacity_mah": 5000,
                "display_size_in": 6.5,
                "screen_to_body_pct": 84,
                "resolution_width_px": 1080,
                "resolution_height_px": 2400,
                "ppi": 405,
                "body_weight_g": 190,
                "fast_charging_w": 25,
                "main_camera_max_mp": 50,
                "network_generation": 5,
                "sensor_count": 5,
                "launch_year": 2020,
                "budget_price_eur": 450,
            },
            {
                "scenario": "B2B_flagship_launch",
                "brand": "Apple",
                "ram_gb": 8,
                "storage_gb": 256,
                "battery_capacity_mah": 4300,
                "display_size_in": 6.7,
                "screen_to_body_pct": 87,
                "resolution_width_px": 1284,
                "resolution_height_px": 2778,
                "ppi": 460,
                "body_weight_g": 228,
                "fast_charging_w": 27,
                "main_camera_max_mp": 48,
                "network_generation": 5,
                "sensor_count": 6,
                "launch_year": 2020,
                "budget_price_eur": 950,
            },
            {
                "scenario": "B2B_value_launch",
                "brand": "Xiaomi",
                "ram_gb": 6,
                "storage_gb": 128,
                "battery_capacity_mah": 5000,
                "display_size_in": 6.4,
                "screen_to_body_pct": 83,
                "resolution_width_px": 1080,
                "resolution_height_px": 2400,
                "ppi": 395,
                "body_weight_g": 195,
                "fast_charging_w": 30,
                "main_camera_max_mp": 64,
                "network_generation": 5,
                "sensor_count": 5,
                "launch_year": 2020,
                "budget_price_eur": 300,
            },
        ]
        business = []
        for spec in business_specs:
            business.append({**spec, **self.business_price_guide(spec)})
        business_df = pd.DataFrame(business)
        business_df.to_csv(self.output_dir / "two_way_business_price_guides.csv", index=False)

        recs = []
        for scenario, budget, segment in [
            ("student_balanced", 200, None),
            ("worker_midrange", 400, "Mid-range"),
            ("premium_flagship", 800, "Flagship"),
            ("entry_second_phone", 150, "Entry"),
        ]:
            rec = self.recommend_for_user(budget, top_n=8, segment=segment)
            rec.insert(0, "scenario", scenario)
            rec.insert(1, "budget_eur", budget)
            recs.append(rec)
        rec_df = pd.concat(recs, ignore_index=True)
        rec_df.to_csv(self.output_dir / "two_way_user_recommendations.csv", index=False)
        return business_df, rec_df


def parse_args() -> argparse.Namespace:
    """Parse command-line options for demo, business, and recommendation modes.

    Separate modes make the script usable for both stakeholders:
    - demo: regenerate sample modeling outputs,
    - business: predict a fair price from planned specs,
    - recommend: return budget-safe user recommendations.
    """
    parser = argparse.ArgumentParser(description="DS Team 7 GSM two-way smartphone solution")
    parser.add_argument("--mode", choices=["demo", "business", "recommend"], default="demo")
    parser.add_argument("--brand", default=None)
    parser.add_argument("--budget-eur", type=float, default=None)
    parser.add_argument("--budget-price-eur", type=float, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--segment", default=None)
    parser.add_argument("--ram-gb", type=float, default=None)
    parser.add_argument("--storage-gb", type=float, default=None)
    parser.add_argument("--spec-score", type=float, default=None)
    parser.add_argument("--battery-capacity-mah", type=float, default=None)
    parser.add_argument("--main-camera-mp", type=float, default=None)
    parser.add_argument("--network-generation", type=float, default=None)
    parser.add_argument("--launch-year", type=float, default=None)
    parser.add_argument("--phone-age", type=float, default=None)
    parser.add_argument("--display-size-in", type=float, default=None)
    parser.add_argument("--ppi", type=float, default=None)
    parser.add_argument("--resolution-width-px", type=float, default=None)
    parser.add_argument("--resolution-height-px", type=float, default=None)
    parser.add_argument("--screen-to-body-pct", type=float, default=None)
    parser.add_argument("--body-weight-g", type=float, default=None)
    parser.add_argument("--fast-charging-w", type=float, default=None)
    parser.add_argument("--sensor-count", type=float, default=None)
    parser.add_argument("--json", default=None)
    return parser.parse_args()


def main() -> int:
    """Run the selected CLI mode and print a readable result.

    The function returns an integer exit code so it can be used both as a script and as a simple command-line tool in Colab or a terminal.
    """
    args = parse_args()
    solution = TwoWaySmartphoneSolution()
    if args.mode == "demo":
        # Demo mode writes both B2B and B2C example outputs.
        # This is the fastest way to confirm that saved model artifacts and output CSV files connect correctly.
        business, recs = solution.run_demo()
        print("\n[B2B/B2C GSM demo complete]")
        print("\nBusiness price guide examples")
        print(
            business[
                ["scenario", "brand", "predicted_price_eur", "budget_price_eur", "guidance", "price_gap_eur"]
            ].to_string(index=False)
        )
        print("\nUser recommendation examples")
        print(recs.head(12).to_string(index=False))
        return 0

    if args.mode == "business":
        # Business mode accepts either a JSON spec dictionary or individual CLI fields.
        # JSON is useful for reproducible tests; CLI fields are easier for quick manual experiments.
        specs = json.loads(args.json) if args.json else {
            "brand": args.brand,
            "budget_price_eur": args.budget_price_eur,
            "ram_gb": args.ram_gb,
            "storage_gb": args.storage_gb,
            "spec_score_0_100": args.spec_score,
            "battery_capacity_mah": args.battery_capacity_mah,
            "main_camera_max_mp": args.main_camera_mp,
            "network_generation": args.network_generation,
            "launch_year": args.launch_year,
            "phone_age": args.phone_age,
            "display_size_in": args.display_size_in,
            "ppi": args.ppi,
            "resolution_width_px": args.resolution_width_px,
            "resolution_height_px": args.resolution_height_px,
            "screen_to_body_pct": args.screen_to_body_pct,
            "body_weight_g": args.body_weight_g,
            "fast_charging_w": args.fast_charging_w,
            "sensor_count": args.sensor_count,
        }
        print(json.dumps(solution.business_price_guide(specs), ensure_ascii=False, indent=2))
        return 0

    budget = args.budget_eur
    if budget is None:
        # Direct budget input satisfies the project requirement that users can enter their own budget constraint for recommendation.
        budget = float(input("사용자 예산을 EUR 단위로 입력하세요: ").strip())
    recs = solution.recommend_for_user(
        budget_eur=budget,
        top_n=args.top_n,
        segment=args.segment,
        brand=args.brand,
    )
    print(recs.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
