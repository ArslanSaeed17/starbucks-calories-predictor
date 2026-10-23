"""
Starbucks Calories Predictor
============================
A Streamlit dashboard that explores the Starbucks drink menu nutrition data and
estimates a drink's calories with a Decision Tree model.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud (main file path: app.py)

Independent educational project. Not affiliated with or endorsed by Starbucks.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

# --------------------------------------------------------------------------- #
# Page configuration (must be the first Streamlit call)
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Starbucks Calories Predictor",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
APP_DIR = Path(__file__).resolve().parent
DATA_FILENAME = "starbucks_drinkMenu_expanded.csv"

TARGET = "Calories"
NAME_COL = "Beverage"
CATEGORY_COL = "Beverage_category"
PREP_COL = "Beverage_prep"
REQUIRED_COLUMNS = [CATEGORY_COL, NAME_COL, PREP_COL, TARGET]
# The model only sees these two columns. Other nutrition columns are NOT used
# as inputs because they would leak the answer (calories come from fat, carbs
# and protein), which would make the task trivial.
FEATURE_COLUMNS = [CATEGORY_COL, PREP_COL]

# Nutrition columns we try to use if they exist (names are whitespace-normalised).
NUTRIENT_COLUMNS = [
    "Total Fat (g)",
    "Trans Fat (g)",
    "Saturated Fat (g)",
    "Sodium (mg)",
    "Total Carbohydrates (g)",
    "Cholesterol (mg)",
    "Dietary Fibre (g)",
    "Sugars (g)",
    "Protein (g)",
    "Vitamin A (% DV)",
    "Vitamin C (% DV)",
    "Calcium (% DV)",
    "Iron (% DV)",
    "Caffeine (mg)",
]
GRAM_COLUMNS_WANTED = [
    "Total Fat (g)",
    "Saturated Fat (g)",
    "Total Carbohydrates (g)",
    "Sugars (g)",
    "Protein (g)",
    "Dietary Fibre (g)",
]
HIGHLIGHT_COLUMNS_WANTED = [
    "Total Fat (g)",
    "Total Carbohydrates (g)",
    "Sugars (g)",
    "Protein (g)",
    "Sodium (mg)",
    "Caffeine (mg)",
]
SIZE_ORDER = ["Short", "Tall", "Grande", "Venti", "Solo", "Doppio"]
MIN_ROWS_FOR_MODEL = 20

# Palette
DARK_GREEN = "#1E3932"
GREEN = "#00704A"
LIGHT_GREEN = "#A8D5BA"
CREAM = "#F7F3EA"
GOLD = "#CBA258"


class DataError(Exception):
    """Raised when the dataset is missing or not usable."""


# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
CUSTOM_CSS = """
<style>
.stApp { background: #F7F3EA; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1200px; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Text colours */
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #1E3932; font-family: Georgia, 'Times New Roman', serif;
}
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stWidgetLabel"] p,
.stApp label { color: #1E3932; }
.stApp [data-testid="stCaptionContainer"] { color: #5F7A70; }

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E3932 0%, #0F2A23 100%);
}
section[data-testid="stSidebar"] * { color: #F2EEE3 !important; }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 0.25rem; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 0.55rem 0.8rem; border-radius: 12px; width: 100%;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255, 255, 255, 0.10);
}
.sb-brand { font-family: Georgia, serif; font-size: 1.35rem; font-weight: 700; margin-bottom: 0.2rem; }
.sb-brand-sub { font-size: 0.8rem; opacity: 0.8; margin-bottom: 1rem; }
.sb-side-note { font-size: 0.75rem; opacity: 0.75; line-height: 1.4; }

/* Hero */
.sb-hero {
    background: linear-gradient(135deg, #1E3932 0%, #00704A 100%);
    border-radius: 22px; padding: 1.9rem 2.2rem; margin-bottom: 1.4rem;
    box-shadow: 0 10px 30px rgba(30, 57, 50, 0.25); border-bottom: 4px solid #CBA258;
}
.stApp .sb-hero .sb-hero-title {
    font-family: Georgia, 'Times New Roman', serif; font-size: 2.1rem; font-weight: 700;
    color: #FFF8E7; margin: 0; line-height: 1.2;
}
.stApp .sb-hero .sb-hero-sub { color: #DCE8E1; margin-top: 0.45rem; font-size: 1.02rem; }
.stApp .sb-hero .sb-badge {
    display: inline-block; margin-top: 0.9rem; padding: 0.25rem 0.8rem; border-radius: 999px;
    background: rgba(203, 162, 88, 0.2); border: 1px solid #CBA258; color: #F3DFB3; font-size: 0.78rem;
}

/* KPI cards */
.sb-kpi {
    background: #FFFFFF; border-radius: 18px; padding: 1.1rem 1.2rem; min-height: 130px;
    box-shadow: 0 4px 16px rgba(30, 57, 50, 0.10); border-left: 5px solid #00704A;
}
.sb-kpi.gold { border-left-color: #CBA258; }
.sb-kpi-label { font-size: 0.74rem; letter-spacing: 0.08em; text-transform: uppercase; color: #6B7F77; }
.sb-kpi-value { font-family: Georgia, serif; font-size: 1.9rem; font-weight: 700; color: #1E3932; margin-top: 0.2rem; }
.sb-kpi-value.sm { font-size: 1.05rem; line-height: 1.3; }
.sb-kpi-sub { font-size: 0.85rem; color: #8A6D2F; margin-top: 0.25rem; }

/* Result card */
.stApp .sb-result {
    background: linear-gradient(135deg, #1E3932, #00704A); border-radius: 22px; padding: 1.8rem 1.4rem;
    text-align: center; box-shadow: 0 10px 28px rgba(30, 57, 50, 0.28); border: 2px solid #CBA258;
}
.stApp .sb-result .sb-result-label { color: #DCE8E1; font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase; }
.stApp .sb-result .sb-result-number { font-family: Georgia, serif; font-size: 4rem; font-weight: 700; color: #FFF8E7; line-height: 1.1; }
.stApp .sb-result .sb-result-unit { color: #CBA258; font-size: 1.1rem; font-weight: 600; }
.stApp .sb-result .sb-result-sub { color: #DCE8E1; margin-top: 0.7rem; font-size: 0.95rem; }

/* Empty state & disclaimer */
.sb-empty {
    background: #FFFFFF; border: 2px dashed #A8D5BA; border-radius: 22px; padding: 2.2rem 1.4rem;
    text-align: center; color: #5F7A70;
}
.sb-disclaimer {
    background: #FFF6DD; border-left: 5px solid #CBA258; border-radius: 12px;
    padding: 0.9rem 1.1rem; color: #5B4A1F; font-size: 0.92rem; margin: 0.8rem 0;
}

/* Buttons */
.stApp .stButton > button {
    background: #00704A; border: none; border-radius: 999px; padding: 0.65rem 1.4rem;
    font-weight: 600; box-shadow: 0 4px 12px rgba(0, 112, 74, 0.3);
}
.stApp .stButton > button, .stApp .stButton > button p { color: #FFFFFF !important; }
.stApp .stButton > button:hover { background: #1E3932; }

/* Bordered containers and metrics */
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 18px; }
[data-testid="stMetricValue"] { color: #00704A; font-family: Georgia, serif; }

/* Mobile */
@media (max-width: 640px) {
    .sb-hero { padding: 1.2rem 1.2rem; }
    .stApp .sb-hero .sb-hero-title { font-size: 1.5rem; }
    .stApp .sb-result .sb-result-number { font-size: 3rem; }
    .sb-kpi-value { font-size: 1.5rem; }
    .sb-kpi { min-height: 0; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(
        """
        <div class="sb-hero">
            <div class="sb-hero-title">☕ Starbucks Calories Predictor</div>
            <div class="sb-hero-sub">Explore the drink menu's nutrition data and estimate calories with machine learning.</div>
            <span class="sb-badge">Independent student project · not affiliated with Starbucks</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "", small: bool = False, gold: bool = False) -> str:
    value_class = "sb-kpi-value sm" if small else "sb-kpi-value"
    card_class = "sb-kpi gold" if gold else "sb-kpi"
    return (
        f'<div class="{card_class}">'
        f'<div class="sb-kpi-label">{html.escape(label)}</div>'
        f'<div class="{value_class}">{html.escape(value)}</div>'
        f'<div class="sb-kpi-sub">{html.escape(sub)}</div>'
        f"</div>"
    )


def disclaimer_box(text: str) -> None:
    st.markdown(f'<div class="sb-disclaimer">{html.escape(text)}</div>', unsafe_allow_html=True)


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Helvetica, Arial, sans-serif", color=DARK_GREEN),
        title=dict(font=dict(size=17, family="Georgia, serif")),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="left", x=0),
    )
    return fig


# --------------------------------------------------------------------------- #
# Data loading and cleaning
# --------------------------------------------------------------------------- #
def find_dataset() -> Path | None:
    """Look for the CSV next to app.py (exact name first, then a similar name)."""
    exact = APP_DIR / DATA_FILENAME
    if exact.exists():
        return exact
    matches = sorted(APP_DIR.glob("*starbucks_drinkMenu_expanded*.csv"))
    return matches[0] if matches else None


def to_number(series: pd.Series) -> pd.Series:
    """Convert messy text such as '12%', ' 3.5 ' or 'Varies' to numbers (invalid -> NaN)."""
    cleaned = series.astype(str).str.strip().str.replace("%", "", regex=False)
    return pd.to_numeric(cleaned, errors="coerce")


@st.cache_data(show_spinner="Loading Starbucks dataset...")
def load_data() -> tuple[pd.DataFrame, list[str]]:
    path = find_dataset()
    if path is None:
        raise DataError(
            f"Dataset not found. Please place your original **{DATA_FILENAME}** in the same "
            "folder as `app.py` (for Streamlit Community Cloud: in the repository root) and reload the app."
        )

    try:
        try:
            raw = pd.read_csv(path)
        except UnicodeDecodeError:
            raw = pd.read_csv(path, encoding="latin-1")
    except Exception as exc:  # noqa: BLE001 - show a friendly message instead of a traceback
        raise DataError(f"The dataset file `{path.name}` could not be read: {exc}") from exc

    # Column names in this CSV contain stray spaces (e.g. ' Total Fat (g)') - normalise them.
    raw.columns = [" ".join(str(col).split()) for col in raw.columns]

    missing = [col for col in REQUIRED_COLUMNS if col not in raw.columns]
    if missing:
        raise DataError(
            "The dataset is missing required column(s): "
            + ", ".join(f"`{m}`" for m in missing)
            + ". Columns found: "
            + ", ".join(f"`{c}`" for c in raw.columns)
        )

    df = raw.copy()
    for col in (NAME_COL, CATEGORY_COL, PREP_COL):
        df[col] = df[col].astype("object")
        df[col] = df[col].where(df[col].notna(), np.nan)
        df[col] = df[col].map(lambda v: " ".join(str(v).split()) if isinstance(v, str) else v)
        df[col] = df[col].replace("", np.nan)

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[NAME_COL, TARGET]).copy()

    nutrients = [col for col in NUTRIENT_COLUMNS if col in df.columns]
    for col in nutrients:
        df[col] = to_number(df[col])

    df = df.drop_duplicates().reset_index(drop=True)
    if df.empty:
        raise DataError("The dataset has no usable rows after cleaning (Beverage and Calories are required).")

    # A readable, unique label for each menu row.
    prep_text = df[PREP_COL].fillna("Standard")
    label = df[NAME_COL].astype(str) + " (" + prep_text.astype(str) + ")"
    duplicated = label.duplicated(keep=False)
    label = label.where(~duplicated, label + " [" + df[CATEGORY_COL].fillna("Unknown").astype(str) + "]")
    still_duplicated = label.duplicated(keep=False)
    label = label.where(~still_duplicated, label + " #" + (df.index + 1).astype(str))
    df["Label"] = label

    return df, nutrients


# --------------------------------------------------------------------------- #
# Machine learning
# --------------------------------------------------------------------------- #
def build_pipeline() -> Pipeline:
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        transformers=[("categorical", categorical, FEATURE_COLUMNS)],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", DecisionTreeRegressor(random_state=42)),
        ]
    )


@st.cache_resource(show_spinner="Training the calorie model...")
def train_model(df: pd.DataFrame) -> dict:
    """Train a DecisionTreeRegressor on category + preparation only (no leakage)."""
    data = df[FEATURE_COLUMNS + [TARGET]].copy()
    x_all = data[FEATURE_COLUMNS]
    y_all = data[TARGET].astype(float)

    x_train, x_test, y_train, y_test = train_test_split(x_all, y_all, test_size=0.2, random_state=42)

    pipeline = build_pipeline()
    best_params: dict = {}
    if len(x_train) >= 30:
        # Small, quick search on the training data only; the test set stays untouched.
        search = GridSearchCV(
            pipeline,
            param_grid={
                "model__max_depth": [4, 6, 8, 10, None],
                "model__min_samples_leaf": [1, 2, 3, 5],
            },
            cv=5,
            scoring="neg_mean_absolute_error",
        )
        search.fit(x_train, y_train)
        model = search.best_estimator_
        best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    else:
        model = pipeline.fit(x_train, y_train)

    predictions = np.clip(model.predict(x_test), 0, None)
    baseline = np.full(len(y_test), float(y_train.mean()))

    # Final model: refit the chosen settings on all rows so every menu item is learned.
    final_model = build_pipeline()
    if best_params:
        final_model.set_params(**{f"model__{k}": v for k, v in best_params.items()})
    final_model.fit(x_all, y_all)

    return {
        "model": final_model,
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)) if len(y_test) > 1 else float("nan"),
        "baseline_mae": float(mean_absolute_error(y_test, baseline)),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "best_params": best_params,
        "y_test": y_test.to_numpy(),
        "y_pred": predictions,
    }


def predict_calories(model: Pipeline, category: str, prep: str) -> int:
    # Column names and order are identical to the training data (FEATURE_COLUMNS).
    row = pd.DataFrame([{CATEGORY_COL: category, PREP_COL: prep}], columns=FEATURE_COLUMNS)
    value = float(model.predict(row)[0])
    return int(max(0, round(value)))


def get_model_info(df: pd.DataFrame) -> tuple[dict | None, str | None]:
    if len(df) < MIN_ROWS_FOR_MODEL:
        return None, (
            f"The dataset has only {len(df)} usable rows. At least {MIN_ROWS_FOR_MODEL} are needed "
            "to train and evaluate the model reliably."
        )
    try:
        return train_model(df), None
    except Exception as exc:  # noqa: BLE001
        return None, f"The model could not be trained: {exc}"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def split_label(column: str) -> tuple[str, str]:
    match = re.match(r"^(.*?)\s*\((.*)\)$", column)
    if match:
        return match.group(1), match.group(2)
    return column, ""


def format_value(value: float | None, unit: str = "") -> str:
    if value is None or pd.isna(value):
        return "n/a"
    text = f"{value:,.1f}".rstrip("0").rstrip(".") if isinstance(value, float) else f"{value:,}"
    return f"{text} {unit}".strip()


def sort_preps(preps: list[str]) -> list[str]:
    def key(prep: str) -> tuple[int, str]:
        first = prep.split()[0] if prep.split() else prep
        return (SIZE_ORDER.index(first) if first in SIZE_ORDER else 99, prep)

    return sorted(preps, key=key)


def available(columns: list[str], df: pd.DataFrame) -> list[str]:
    return [c for c in columns if c in df.columns]


def metric_grid(items: list[tuple[str, str]], per_row: int = 4) -> None:
    for start in range(0, len(items), per_row):
        chunk = items[start : start + per_row]
        cols = st.columns(per_row)
        for col, (label, value) in zip(cols, chunk):
            col.metric(label, value)


# --------------------------------------------------------------------------- #
# Page: Dashboard
# --------------------------------------------------------------------------- #
def render_dashboard(df: pd.DataFrame) -> None:
    st.subheader("Dashboard")
    st.caption("A quick overview of the whole drink menu. Every number is calculated from the dataset.")

    lowest = df.loc[df[TARGET].idxmin()]
    highest = df.loc[df[TARGET].idxmax()]
    low_ties = int((df[TARGET] == lowest[TARGET]).sum()) - 1
    high_ties = int((df[TARGET] == highest[TARGET]).sum()) - 1

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(
        kpi_card("Total beverages", f"{len(df):,}", f"{df[NAME_COL].nunique()} unique drink names"),
        unsafe_allow_html=True,
    )
    c2.markdown(
        kpi_card("Average calories", f"{df[TARGET].mean():.0f} kcal", f"Median {df[TARGET].median():.0f} kcal", gold=True),
        unsafe_allow_html=True,
    )
    low_sub = f"{lowest[TARGET]:.0f} kcal" + (f" · +{low_ties} more tied" if low_ties > 0 else "")
    c3.markdown(kpi_card("Lowest-calorie drink", str(lowest[NAME_COL]), low_sub, small=True), unsafe_allow_html=True)
    high_sub = f"{highest[TARGET]:.0f} kcal" + (f" · +{high_ties} more tied" if high_ties > 0 else "")
    c4.markdown(
        kpi_card("Highest-calorie drink", str(highest[NAME_COL]), high_sub, small=True, gold=True),
        unsafe_allow_html=True,
    )

    st.write("")
    left, right = st.columns(2)
    with left:
        fig = px.histogram(df, x=TARGET, nbins=25, title="Calorie distribution", color_discrete_sequence=[GREEN])
        fig.add_vline(
            x=df[TARGET].mean(),
            line_dash="dash",
            line_color=GOLD,
            annotation_text=f"Average {df[TARGET].mean():.0f}",
            annotation_position="top right",
        )
        fig.update_layout(xaxis_title="Calories (kcal)", yaxis_title="Number of drinks", bargap=0.05)
        st.plotly_chart(style_fig(fig), use_container_width=True)
    with right:
        by_category = (
            df.dropna(subset=[CATEGORY_COL]).groupby(CATEGORY_COL)[TARGET].mean().sort_values().reset_index()
        )
        fig = px.bar(
            by_category,
            x=TARGET,
            y=CATEGORY_COL,
            orientation="h",
            title="Average calories by beverage category",
            text=by_category[TARGET].round(0).astype(int),
            color_discrete_sequence=[GREEN],
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_title="Average calories (kcal)", yaxis_title="", yaxis=dict(automargin=True))
        st.plotly_chart(style_fig(fig), use_container_width=True)

    left, right = st.columns(2)
    ordered = df.sort_values([TARGET, "Label"], ascending=[False, True])
    top5 = ordered.head(5)
    bottom5 = df.sort_values([TARGET, "Label"], ascending=[True, True]).head(5)
    for column, data, title, color in (
        (left, top5, "Top 5 highest-calorie drinks", GOLD),
        (right, bottom5, "Bottom 5 lowest-calorie drinks", LIGHT_GREEN),
    ):
        with column:
            fig = px.bar(
                data,
                x=TARGET,
                y="Label",
                orientation="h",
                title=title,
                text=TARGET,
                hover_data={CATEGORY_COL: True, "Label": False},
                color_discrete_sequence=[color],
            )
            fig.update_traces(textposition="outside", cliponaxis=False, marker_line_color=DARK_GREEN, marker_line_width=1)
            fig.update_layout(
                xaxis_title="Calories (kcal)",
                yaxis_title="",
                yaxis=dict(autorange="reversed", automargin=True),
            )
            st.plotly_chart(style_fig(fig), use_container_width=True)
    st.caption("Some drinks share the same calorie value, so the lists above show a sample of the tied drinks.")

    gram_cols = available(GRAM_COLUMNS_WANTED, df)
    if gram_cols and df[CATEGORY_COL].notna().any():
        macro = df.dropna(subset=[CATEGORY_COL]).groupby(CATEGORY_COL)[gram_cols].mean().reset_index()
        macro = macro.melt(id_vars=CATEGORY_COL, var_name="Nutrient", value_name="Grams")
        macro["Nutrient"] = macro["Nutrient"].map(lambda c: split_label(c)[0])
        fig = px.bar(
            macro,
            x=CATEGORY_COL,
            y="Grams",
            color="Nutrient",
            barmode="group",
            title="Nutrition comparison: average grams per drink by category",
            color_discrete_sequence=[DARK_GREEN, GREEN, LIGHT_GREEN, GOLD, "#8C6B3F", "#6F9E8B"],
        )
        fig.update_layout(xaxis_title="", yaxis_title="Average grams", xaxis=dict(automargin=True))
        st.plotly_chart(style_fig(fig, height=430), use_container_width=True)


# --------------------------------------------------------------------------- #
# Page: Calorie Predictor
# --------------------------------------------------------------------------- #
def render_predictor(df: pd.DataFrame, model_info: dict | None, model_error: str | None) -> None:
    st.subheader("Calorie Predictor")
    st.caption("Pick a beverage category and preparation. The model estimates calories from those two choices.")

    if model_info is None:
        st.warning(model_error or "The prediction model is not available.")
        return

    categories = sorted(df[CATEGORY_COL].dropna().unique().tolist())
    if not categories:
        st.error("The dataset has no beverage categories to choose from.")
        return

    form_col, result_col = st.columns([1, 1.15], gap="large")

    with form_col:
        with st.container(border=True):
            st.markdown("#### Build your drink")
            category = st.selectbox(
                "Beverage category",
                categories,
                help="Categories come straight from the dataset.",
            )
            preps = sort_preps(df.loc[df[CATEGORY_COL] == category, PREP_COL].dropna().unique().tolist())
            if not preps:
                st.warning("This category has no preparation options in the dataset. Choose another category.")
            prep = st.selectbox(
                "Preparation (size or milk)",
                preps,
                help="Only preparations that exist for the chosen category are listed.",
                disabled=not preps,
            )
            clicked = st.button("Predict Calories", type="primary", use_container_width=True, disabled=not preps)

        if clicked and preps:
            try:
                value = predict_calories(model_info["model"], category, prep)
                st.session_state["prediction"] = {"category": category, "prep": prep, "value": value}
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")

    prediction = st.session_state.get("prediction")
    # Ignore a stored prediction if it no longer matches the dataset's options.
    if prediction and prediction["category"] not in categories:
        prediction = None

    with result_col:
        if prediction is None:
            st.markdown(
                '<div class="sb-empty"><div style="font-size:2.4rem">☕</div>'
                "<div><b>No prediction yet</b></div>"
                "<div>Choose a category and preparation, then press <b>Predict Calories</b>.</div></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="sb-result">'
                '<div class="sb-result-label">Estimated calories</div>'
                f'<div class="sb-result-number">{prediction["value"]}</div>'
                '<div class="sb-result-unit">kcal</div>'
                f'<div class="sb-result-sub">{html.escape(prediction["category"])} · {html.escape(prediction["prep"])}</div>'
                f'<div class="sb-result-sub">Typical error on unseen test drinks: about ±{model_info["mae"]:.0f} kcal</div>'
                "</div>",
                unsafe_allow_html=True,
            )

    disclaimer_box(
        "Disclaimer: this is a machine-learning estimate based on category and preparation only. "
        "It is not an official Starbucks nutrition value. Check Starbucks' official information for exact numbers."
    )

    if prediction is not None:
        render_prediction_context(df, prediction)

    with st.expander("How reliable is the model?"):
        render_model_report(model_info)


def render_prediction_context(df: pd.DataFrame, prediction: dict) -> None:
    st.markdown("#### Nutrition context from the dataset")
    group = df[(df[CATEGORY_COL] == prediction["category"]) & (df[PREP_COL] == prediction["prep"])]
    overall_avg = df[TARGET].mean()
    diff = prediction["value"] - overall_avg
    direction = "above" if diff >= 0 else "below"
    st.write(
        f"The estimate of **{prediction['value']} kcal** is about **{abs(diff):.0f} kcal {direction}** "
        f"the menu average of {overall_avg:.0f} kcal."
    )

    if group.empty:
        st.info("The dataset has no drinks with exactly this category and preparation, so no group statistics are shown.")
        return

    metric_grid(
        [
            ("Drinks in this group", f"{len(group)}"),
            ("Group average", f"{group[TARGET].mean():.0f} kcal"),
            ("Group lowest", f"{group[TARGET].min():.0f} kcal"),
            ("Group highest", f"{group[TARGET].max():.0f} kcal"),
        ]
    )

    gram_cols = available(GRAM_COLUMNS_WANTED, df)
    if gram_cols:
        rows = []
        for col in gram_cols:
            name = split_label(col)[0]
            rows.append({"Nutrient": name, "Grams": group[col].mean(), "Group": "This category + preparation"})
            rows.append({"Nutrient": name, "Grams": df[col].mean(), "Group": "All drinks"})
        chart_df = pd.DataFrame(rows).dropna(subset=["Grams"])
        if not chart_df.empty:
            fig = px.bar(
                chart_df,
                x="Nutrient",
                y="Grams",
                color="Group",
                barmode="group",
                title="Average nutrition: this selection vs. all drinks",
                color_discrete_map={"This category + preparation": GREEN, "All drinks": GOLD},
            )
            fig.update_layout(xaxis_title="", yaxis_title="Average grams")
            st.plotly_chart(style_fig(fig), use_container_width=True)

    with st.expander(f"Drinks in this group ({len(group)})"):
        show_cols = [NAME_COL, TARGET] + available(HIGHLIGHT_COLUMNS_WANTED, df)
        st.dataframe(
            group[show_cols].sort_values(TARGET, ascending=False),
            use_container_width=True,
            hide_index=True,
        )


def render_model_report(model_info: dict) -> None:
    metric_grid(
        [
            ("MAE (test set)", f"{model_info['mae']:.1f} kcal"),
            ("R² (test set)", "n/a" if pd.isna(model_info["r2"]) else f"{model_info['r2']:.2f}"),
            ("Baseline MAE (always guess the average)", f"{model_info['baseline_mae']:.1f} kcal"),
            ("Train / test rows", f"{model_info['n_train']} / {model_info['n_test']}"),
        ],
        per_row=2,
    )
    st.write(
        "**MAE** is the average size of the model's mistakes in calories (lower is better). "
        "**R²** shows how much of the calorie variation the model explains (1.0 is perfect, 0 is no better than "
        "guessing the average). These numbers were computed on drinks the model did not see while training, "
        "and they will change if the dataset changes."
    )
    st.write(
        "Calories also depend on the specific drink (for example syrups and toppings), which this model "
        "deliberately does not see. That is why estimates are approximate."
    )
    if model_info["best_params"]:
        st.caption(f"Decision tree settings chosen by cross-validation: {model_info['best_params']}")

    y_test, y_pred = model_info["y_test"], model_info["y_pred"]
    if len(y_test) > 0:
        top = float(max(np.max(y_test), np.max(y_pred)))
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=y_test,
                y=y_pred,
                mode="markers",
                name="Test drinks",
                marker=dict(color=GREEN, size=9, opacity=0.65, line=dict(color=DARK_GREEN, width=1)),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, top], y=[0, top], mode="lines", name="Perfect prediction", line=dict(color=GOLD, dash="dash")
            )
        )
        fig.update_layout(
            title="Actual vs. predicted calories (test set)",
            xaxis_title="Actual calories",
            yaxis_title="Predicted calories",
        )
        st.plotly_chart(style_fig(fig, height=400), use_container_width=True)


# --------------------------------------------------------------------------- #
# Page: Nutrition Explorer
# --------------------------------------------------------------------------- #
def render_explorer(df: pd.DataFrame, nutrients: list[str]) -> None:
    st.subheader("Nutrition Explorer")
    st.caption("Search, filter and compare the real nutrition facts from the dataset.")

    categories = sorted(df[CATEGORY_COL].dropna().unique().tolist())
    preps = sort_preps(df[PREP_COL].dropna().unique().tolist())

    with st.container(border=True):
        f1, f2, f3, f4 = st.columns([1.4, 1.2, 1.2, 1])
        query = f1.text_input("Search by drink name", placeholder="e.g. mocha, latte, tea", help="Not case sensitive.")
        chosen_categories = f2.multiselect("Category", categories, placeholder="All categories")
        chosen_preps = f3.multiselect("Preparation", preps, placeholder="All preparations")
        sort_choice = f4.selectbox("Sort by", ["Calories: high to low", "Calories: low to high", "Name: A to Z"])

    filtered = df.copy()
    if query.strip():
        filtered = filtered[filtered[NAME_COL].astype(str).str.contains(query.strip(), case=False, na=False, regex=False)]
    if chosen_categories:
        filtered = filtered[filtered[CATEGORY_COL].isin(chosen_categories)]
    if chosen_preps:
        filtered = filtered[filtered[PREP_COL].isin(chosen_preps)]

    if sort_choice == "Calories: high to low":
        filtered = filtered.sort_values([TARGET, NAME_COL], ascending=[False, True])
    elif sort_choice == "Calories: low to high":
        filtered = filtered.sort_values([TARGET, NAME_COL], ascending=[True, True])
    else:
        filtered = filtered.sort_values([NAME_COL, PREP_COL])

    if filtered.empty:
        st.info("No drinks match your search and filters. Try a different name or clear some filters.")
        return

    st.write(f"**{len(filtered)}** of {len(df)} drinks shown.")
    table_cols = [NAME_COL, CATEGORY_COL, PREP_COL, TARGET] + nutrients
    table = filtered[table_cols].rename(
        columns={NAME_COL: "Beverage", CATEGORY_COL: "Category", PREP_COL: "Preparation"}
    )
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        height=380,
        column_config={
            TARGET: st.column_config.ProgressColumn(
                "Calories", min_value=0, max_value=int(df[TARGET].max()), format="%d"
            )
        },
    )

    labels = filtered["Label"].tolist()

    st.markdown("#### Drink details")
    selected_label = st.selectbox("Choose a drink to inspect", labels, key="explorer_detail")
    selected = filtered[filtered["Label"] == selected_label].iloc[0]
    st.write(f"**{selected[NAME_COL]}** · {selected[CATEGORY_COL] if pd.notna(selected[CATEGORY_COL]) else 'Unknown category'}")

    highlight = available(HIGHLIGHT_COLUMNS_WANTED, df)
    items = [("Calories", f"{selected[TARGET]:.0f} kcal")]
    for col in highlight:
        name, unit = split_label(col)
        items.append((name, format_value(selected[col], unit)))
    metric_grid(items)

    gram_cols = available(GRAM_COLUMNS_WANTED, df)
    same_category = df[df[CATEGORY_COL] == selected[CATEGORY_COL]] if pd.notna(selected[CATEGORY_COL]) else df
    if gram_cols:
        rows = []
        for col in gram_cols:
            name = split_label(col)[0]
            rows.append({"Nutrient": name, "Grams": selected[col], "Compared with": "This drink"})
            rows.append({"Nutrient": name, "Grams": same_category[col].mean(), "Compared with": "Category average"})
        chart_df = pd.DataFrame(rows).dropna(subset=["Grams"])
        if not chart_df.empty:
            fig = px.bar(
                chart_df,
                x="Nutrient",
                y="Grams",
                color="Compared with",
                barmode="group",
                title="This drink vs. its category average (grams)",
                color_discrete_map={"This drink": GREEN, "Category average": GOLD},
            )
            fig.update_layout(xaxis_title="", yaxis_title="Grams")
            st.plotly_chart(style_fig(fig, height=340), use_container_width=True)

    with st.expander("All nutrition facts for this drink"):
        facts = [{"Nutrient": "Calories", "Value": f"{selected[TARGET]:.0f} kcal"}]
        for col in nutrients:
            name, unit = split_label(col)
            facts.append({"Nutrient": name, "Value": format_value(selected[col], unit)})
        st.dataframe(pd.DataFrame(facts), use_container_width=True, hide_index=True)
        st.caption("'n/a' means the dataset has no numeric value for this item (for example 'Varies').")

    st.markdown("#### Compare drinks")
    compare_labels = st.multiselect(
        "Pick 2 to 6 drinks to compare",
        labels,
        max_selections=6,
        placeholder="Choose drinks from the filtered list",
        key="explorer_compare",
    )
    if len(compare_labels) < 2:
        st.info("Select at least two drinks above to see a side-by-side comparison.")
        return

    compare = filtered[filtered["Label"].isin(compare_labels)].copy()
    fig = px.bar(
        compare,
        x="Label",
        y=TARGET,
        text=TARGET,
        title="Calories comparison",
        color_discrete_sequence=[GREEN],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="", yaxis_title="Calories (kcal)", xaxis=dict(automargin=True))
    st.plotly_chart(style_fig(fig, height=360), use_container_width=True)

    if gram_cols:
        long = compare.melt(id_vars="Label", value_vars=gram_cols, var_name="Nutrient", value_name="Grams")
        long["Nutrient"] = long["Nutrient"].map(lambda c: split_label(c)[0])
        long = long.dropna(subset=["Grams"])
        if not long.empty:
            fig = px.bar(
                long,
                x="Nutrient",
                y="Grams",
                color="Label",
                barmode="group",
                title="Nutrient comparison (grams)",
                color_discrete_sequence=[DARK_GREEN, GREEN, LIGHT_GREEN, GOLD, "#8C6B3F", "#6F9E8B"],
            )
            fig.update_layout(xaxis_title="", yaxis_title="Grams", legend_title_text="")
            st.plotly_chart(style_fig(fig, height=420), use_container_width=True)

    side_by_side = compare.set_index("Label")[[TARGET] + nutrients].T
    side_by_side.index = ["Calories"] + [split_label(c)[0] + (f" ({split_label(c)[1]})" if split_label(c)[1] else "") for c in nutrients]
    side_by_side = side_by_side.apply(lambda col: col.map(lambda v: format_value(None if pd.isna(v) else float(v))))
    st.dataframe(side_by_side, use_container_width=True)


# --------------------------------------------------------------------------- #
# Page: About
# --------------------------------------------------------------------------- #
def render_about(df: pd.DataFrame, model_info: dict | None, model_error: str | None) -> None:
    st.subheader("About this project")
    st.write(
        "The **Starbucks Calories Predictor** is an educational data-science project that combines a nutrition "
        "dashboard with a small machine-learning model. It is built with Python, Streamlit, Pandas, Plotly and "
        "scikit-learn."
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Dataset")
        st.write(
            f"The app reads `{DATA_FILENAME}` from the same folder as `app.py`. It currently contains "
            f"**{len(df)} drinks** in **{df[CATEGORY_COL].nunique()} categories**. Column names are tidied "
            "automatically, and text such as 'Varies' is treated as a missing number instead of being guessed."
        )
        st.markdown("#### How the model works")
        st.write(
            "A **Decision Tree Regressor** predicts `Calories` using only **beverage category** and "
            "**preparation**. Missing values are filled with the most common value, categories are one-hot "
            "encoded, and unseen categories are ignored safely. Other nutrition columns are intentionally "
            "excluded to avoid *target leakage*."
        )
    with c2:
        st.markdown("#### Model performance")
        if model_info is None:
            st.warning(model_error or "The model is not available.")
        else:
            render_model_report(model_info)

    st.markdown("#### Limitations")
    st.write(
        "- The model does not know the specific drink, syrups, toppings or custom orders.\n"
        "- Accuracy depends on the dataset, which has only a few hundred rows.\n"
        "- Estimates are for learning and exploration, not dietary or medical advice."
    )
    disclaimer_box(
        "This project is independent and not affiliated with, endorsed by or sponsored by Starbucks. "
        "Predictions are model estimates, not official Starbucks nutrition values."
    )


# --------------------------------------------------------------------------- #
# Navigation and main
# --------------------------------------------------------------------------- #
PAGES = ["🏠 Dashboard", "🔮 Calorie Predictor", "🔍 Nutrition Explorer", "ℹ️ About Project"]


def render_sidebar(df: pd.DataFrame | None) -> str:
    with st.sidebar:
        st.markdown('<div class="sb-brand">☕ Calories Predictor</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-brand-sub">Starbucks-inspired nutrition analytics</div>', unsafe_allow_html=True)
        page = st.radio("Navigation", PAGES, label_visibility="collapsed")
        st.divider()
        if df is not None:
            st.markdown(
                f'<div class="sb-side-note">Dataset: {len(df)} drinks<br>Model: Decision Tree Regressor</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="sb-side-note"><br>Independent project. Not affiliated with Starbucks. '
            "Predictions are estimates, not official nutrition values.</div>",
            unsafe_allow_html=True,
        )
    return page


def main() -> None:
    inject_css()

    df: pd.DataFrame | None = None
    nutrients: list[str] = []
    load_problem: str | None = None
    try:
        df, nutrients = load_data()
    except DataError as exc:
        load_problem = str(exc)
    except Exception as exc:  # noqa: BLE001
        load_problem = f"Something unexpected went wrong while loading the data: {exc}"

    page = render_sidebar(df)
    render_header()

    if df is None:
        st.error(load_problem or "The dataset could not be loaded.")
        st.info(
            "How to fix it: download your original **starbucks_drinkMenu_expanded.csv**, put it next to `app.py`, "
            "and reload this page."
        )
        st.stop()

    if page == PAGES[0]:
        render_dashboard(df)
    elif page == PAGES[1]:
        model_info, model_error = get_model_info(df)
        render_predictor(df, model_info, model_error)
    elif page == PAGES[2]:
        render_explorer(df, nutrients)
    else:
        model_info, model_error = get_model_info(df)
        render_about(df, model_info, model_error)


main()
