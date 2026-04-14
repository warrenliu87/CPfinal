from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from vaultech_analysis.inference import Predictor


st.set_page_config(
    page_title="VaultTech Bath Time Predictor",
    layout="wide",
)

GOLD_PATH = Path("data/gold/pieces.parquet")


@st.cache_resource
def load_predictor() -> Predictor:
    return Predictor()


@st.cache_data
def load_dataset() -> pd.DataFrame:
    if not GOLD_PATH.exists():
        raise FileNotFoundError(f"Gold parquet not found: {GOLD_PATH.resolve()}")

    df = pd.read_parquet(GOLD_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    predictor = load_predictor()
    df["predicted_bath_time_s"] = predictor.predict_batch(df)
    df["prediction_error_s"] = df["predicted_bath_time_s"] - df["lifetime_bath_s"]
    df["abs_prediction_error_s"] = df["prediction_error_s"].abs()

    return df


def build_reference_by_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "lifetime_2nd_strike_s",
        "lifetime_3rd_strike_s",
        "lifetime_4th_strike_s",
        "lifetime_auxiliary_press_s",
        "lifetime_bath_s",
    ]
    return df.groupby("die_matrix")[cols].median()


def safe_diff(a: float | None, b: float | None) -> float | None:
    if pd.isna(a) or pd.isna(b):
        return None
    return float(a - b)


st.title("VaultTech — Bath Time Prediction Dashboard")

df = load_dataset()
reference_by_matrix = build_reference_by_matrix(df)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

die_matrices = sorted(df["die_matrix"].dropna().unique().tolist())
selected_matrices = st.sidebar.multiselect(
    "Die matrix",
    options=die_matrices,
    default=die_matrices,
)

min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()

date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

show_only_slow = st.sidebar.checkbox(
    "Show only slow pieces (P90 per matrix)",
    value=False,
)

# -----------------------------
# Slow-piece thresholds (P90 per matrix)
# -----------------------------
p90_by_matrix = (
    df.groupby("die_matrix")["lifetime_bath_s"]
    .quantile(0.9)
    .to_dict()
)

# -----------------------------
# Apply filters
# -----------------------------
filtered_df = df.copy()

filtered_df = filtered_df[filtered_df["die_matrix"].isin(selected_matrices)]

if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
    filtered_df = filtered_df[
        (filtered_df["timestamp"].dt.date >= start_date) &
        (filtered_df["timestamp"].dt.date <= end_date)
    ]

if show_only_slow:
    def is_slow(row: pd.Series) -> bool:
        threshold = p90_by_matrix.get(row["die_matrix"])
        return threshold is not None and row["lifetime_bath_s"] > threshold

    filtered_df = filtered_df[filtered_df.apply(is_slow, axis=1)]

# -----------------------------
# Summary metrics
# -----------------------------
total_pieces = len(filtered_df)

median_bath_time = (
    filtered_df["lifetime_bath_s"].median()
    if total_pieces > 0 else None
)

median_predicted_time = (
    filtered_df["predicted_bath_time_s"].median()
    if total_pieces > 0 else None
)

mae = (
    filtered_df["abs_prediction_error_s"].mean()
    if total_pieces > 0 else None
)

metric_cols = st.columns(4)

metric_cols[0].metric("Total pieces", f"{total_pieces:,}")

metric_cols[1].metric(
    "Median bath time (s)",
    f"{median_bath_time:.2f}" if median_bath_time is not None else "N/A",
)

metric_cols[2].metric(
    "Median predicted time (s)",
    f"{median_predicted_time:.2f}" if median_predicted_time is not None else "N/A",
)

metric_cols[3].metric(
    "MAE (s)",
    f"{mae:.2f}" if mae is not None else "N/A",
)

# -----------------------------
# Dataset info
# -----------------------------
st.subheader("Loaded dataset")
st.write(f"Rows: {len(df):,}")
st.write(f"Columns: {len(df.columns)}")

# -----------------------------
# Pieces table
# -----------------------------
st.subheader("Filtered pieces")

display_cols = [
    "timestamp",
    "piece_id",
    "die_matrix",
    "lifetime_bath_s",
    "predicted_bath_time_s",
    "prediction_error_s",
    "oee_cycle_time_s",
]
display_cols = [c for c in display_cols if c in filtered_df.columns]

table_df = filtered_df[display_cols].copy().reset_index()
table_df = table_df.rename(columns={"index": "_source_index"})

event = st.dataframe(
    table_df.drop(columns=["_source_index"]),
    use_container_width=True,
    height=400,
    selection_mode="single-row",
    on_select="rerun",
    key="pieces_table",
)

selected_row = None
if event and event.selection and event.selection.rows:
    row_pos = event.selection.rows[0]
    source_index = int(table_df.iloc[row_pos]["_source_index"])
    selected_row = filtered_df.loc[source_index]

# -----------------------------
# Piece detail panel
# -----------------------------
if selected_row is not None:
    st.markdown("---")
    st.subheader("Piece detail")

    detail_cols = st.columns(3)
    detail_cols[0].write(f"**Piece ID:** {selected_row.get('piece_id', 'N/A')}")
    detail_cols[1].write(f"**Die matrix:** {selected_row['die_matrix']}")
    detail_cols[2].write(f"**Timestamp:** {selected_row['timestamp']}")

    matrix = int(selected_row["die_matrix"])
    reference = reference_by_matrix.loc[matrix]

    st.markdown("### Cumulative travel times vs reference")

    cumulative_cols = [
        "lifetime_2nd_strike_s",
        "lifetime_3rd_strike_s",
        "lifetime_4th_strike_s",
        "lifetime_auxiliary_press_s",
        "lifetime_bath_s",
    ]

    cumulative_rows = []
    for col in cumulative_cols:
        actual = float(selected_row[col])
        ref = float(reference[col])
        deviation = actual - ref
        cumulative_rows.append({
            "stage": col,
            "actual_s": round(actual, 2),
            "reference_s": round(ref, 2),
            "deviation_s": round(deviation, 2),
        })

    st.dataframe(pd.DataFrame(cumulative_rows), use_container_width=True)

    st.markdown("### Partial times vs reference")

    actual_partials = {
        "partial_furnace_to_2nd_strike_s": float(selected_row["lifetime_2nd_strike_s"]),
        "partial_2nd_to_3rd_strike_s": safe_diff(
            selected_row["lifetime_3rd_strike_s"],
            selected_row["lifetime_2nd_strike_s"],
        ),
        "partial_3rd_to_4th_strike_s": safe_diff(
            selected_row["lifetime_4th_strike_s"],
            selected_row["lifetime_3rd_strike_s"],
        ),
        "partial_4th_strike_to_auxiliary_press_s": safe_diff(
            selected_row["lifetime_auxiliary_press_s"],
            selected_row["lifetime_4th_strike_s"],
        ),
        "partial_auxiliary_press_to_bath_s": safe_diff(
            selected_row["lifetime_bath_s"],
            selected_row["lifetime_auxiliary_press_s"],
        ),
    }

    reference_partials = {
        "partial_furnace_to_2nd_strike_s": float(reference["lifetime_2nd_strike_s"]),
        "partial_2nd_to_3rd_strike_s": safe_diff(
            reference["lifetime_3rd_strike_s"],
            reference["lifetime_2nd_strike_s"],
        ),
        "partial_3rd_to_4th_strike_s": safe_diff(
            reference["lifetime_4th_strike_s"],
            reference["lifetime_3rd_strike_s"],
        ),
        "partial_4th_strike_to_auxiliary_press_s": safe_diff(
            reference["lifetime_auxiliary_press_s"],
            reference["lifetime_4th_strike_s"],
        ),
        "partial_auxiliary_press_to_bath_s": safe_diff(
            reference["lifetime_bath_s"],
            reference["lifetime_auxiliary_press_s"],
        ),
    }

    partial_rows = []
    for segment, actual in actual_partials.items():
        ref = reference_partials[segment]
        deviation = actual - ref if actual is not None and ref is not None else None
        status = "SLOW" if deviation is not None and deviation > 1.0 else "OK"

        partial_rows.append({
            "segment": segment,
            "actual_s": round(actual, 2) if actual is not None else None,
            "reference_s": round(ref, 2) if ref is not None else None,
            "deviation_s": round(deviation, 2) if deviation is not None else None,
            "status": status,
        })

    st.dataframe(pd.DataFrame(partial_rows), use_container_width=True)
st.markdown("### Partial times: actual vs reference")

chart_df = pd.DataFrame({
        "segment": list(actual_partials.keys()),
        "actual_s": list(actual_partials.values()),
        "reference_s": [reference_partials[k] for k in actual_partials.keys()],
    }).set_index("segment")

st.bar_chart(chart_df)