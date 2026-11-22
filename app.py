
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# =========================================================
# 1. PROJECT CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DATA_DIR = BASE_DIR / "data"

DATASET_PATH = DATA_DIR / "credit_card_default_clean.csv"

st.set_page_config(
    page_title="Credit Card Default Risk Explorer",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp {
        color: inherit;
    }
    .stApp p,
    .stApp label,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stCaptionContainer"] {
        opacity: 0.78;
    }
    .stApp h2,
    .stApp h3,
    .stApp h4 {
        color: inherit;
    }
    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }
    .hero {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        gap: 2rem;
        padding: 2.2rem 2.5rem;
        margin-bottom: 1.5rem;
        border-radius: 24px;
        color: #f7fbff;
        background: linear-gradient(120deg, #10243a 0%, #173b52 65%, #176a70 100%);
        box-shadow: 0 18px 48px rgba(16, 36, 58, 0.16);
    }
    .hero-kicker {
        color: #9de2d5;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }
    .hero h1 {
        color: #ffffff;
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1.08;
        margin: 0;
        letter-spacing: -0.035em;
    }
    .hero p {
        color: #d3e2eb;
        font-size: 1rem;
        max-width: 680px;
        margin: 0.8rem 0 0;
    }
    .stApp .hero p {
        color: #d3e2eb !important;
    }
    .hero-badge {
        flex: 0 0 auto;
        padding: 0.65rem 0.9rem;
        border: 1px solid rgba(206, 245, 237, 0.32);
        border-radius: 999px;
        color: #d6fff5;
        background: rgba(255, 255, 255, 0.08);
        font-size: 0.78rem;
        font-weight: 650;
    }
    .section-heading {
        margin: 0.5rem 0 0.15rem;
        color: inherit;
        font-size: 1.2rem;
        font-weight: 700;
    }
    .section-note {
        margin: 0 0 0.75rem;
        opacity: 0.68;
        font-size: 0.9rem;
    }
    .app-footer {
        padding: 1.5rem 0 0.5rem;
        opacity: 0.68;
        text-align: center;
    }
    [data-testid="stForm"],
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: color-mix(in srgb, currentColor 16%, transparent);
        border-radius: 18px;
        background: color-mix(in srgb, currentColor 2%, transparent);
        box-shadow: 0 8px 28px color-mix(in srgb, currentColor 6%, transparent);
    }
    [data-testid="stMetric"] {
        padding: 1rem 1.1rem;
        border: 1px solid color-mix(in srgb, currentColor 16%, transparent);
        border-radius: 14px;
        background: color-mix(in srgb, currentColor 3%, transparent);
    }
    [data-testid="stMetricLabel"] {
        opacity: 0.72;
    }
    .stButton > button,
    [data-testid="stFormSubmitButton"] > button {
        border-radius: 10px;
        font-weight: 650;
        color: #ffffff !important;
    }
    [data-testid="stFormSubmitButton"] button {
        min-height: 3rem;
        background: linear-gradient(100deg, #176a70, #1d7f78);
        border: 0;
        box-shadow: 0 8px 18px rgba(23, 106, 112, 0.18);
    }
    @media (max-width: 700px) {
        .hero {
            align-items: flex-start;
            flex-direction: column;
            padding: 1.5rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 2. LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_assets():

    model_path = ARTIFACTS_DIR / "loan_default_pipeline.joblib"
    info_path = ARTIFACTS_DIR / "model_info.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    if not info_path.exists():
        raise FileNotFoundError(
            f"Model information not found: {info_path}"
        )

    model = joblib.load(model_path)

    with open(info_path, "r", encoding="utf-8") as file:
        info = json.load(file)

    return model, info


# =========================================================
# 3. APPLICATION HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
      <div>
        <div class="hero-kicker">Credit Risk Lab · Model Explorer</div>
        <h1>Credit Card Default<br>Risk Explorer</h1>
        <p>Explore a machine-learning estimate using a hypothetical customer profile
        and historical credit data.</p>
      </div>
      <div class="hero-badge">EDUCATIONAL DEMO</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Profile information is used only for this prediction and is not saved.")


# =========================================================
# 4. LOAD MODEL AND DATA
# =========================================================

try:

    model, info = load_assets()

except Exception as exc:

    st.error(f"Error loading model: {exc}")

    st.info(
        "Check your artifacts folder and run "
        "python train_model.py if necessary."
    )

    st.stop()


# Load the actual dataset
if not DATASET_PATH.exists():

    st.error(
        f"Dataset not found: {DATASET_PATH}"
    )

    st.info(
        "Make sure credit_card_default_clean.csv "
        "is inside your data folder."
    )

    st.stop()


try:

    dataset = pd.read_csv(DATASET_PATH)

except Exception as exc:

    st.error(f"Error reading dataset: {exc}")
    st.stop()


# Get model features from metadata
features = info.get("feature_columns", [])

if not features:

    st.error(
        "No feature_columns found in model_info.json."
    )

    st.stop()


# Get categorical features from metadata
categorical_features = set(
    info.get("categorical_features", [])
)


# Check whether the model features exist in the dataset
missing_features = [
    feature
    for feature in features
    if feature not in dataset.columns
]

if missing_features:

    st.error(
        f"These model features are missing from the dataset: "
        f"{missing_features}"
    )

    st.stop()


# Check categorical feature names
invalid_categorical = (
    categorical_features - set(features)
)

if invalid_categorical:

    st.error(
        f"Invalid categorical features in model_info.json: "
        f"{list(invalid_categorical)}"
    )

    st.stop()


# =========================================================
# 5. USER-FRIENDLY FIELD NAMES
# =========================================================

display_names = {

    "X1": "Credit Limit (NT dollars)",

    "X2": "Gender",

    "X3": "Education Level",

    "X4": "Marital Status",

    "X5": "Customer Age",

    "X6": "Repayment Status - Previous Month",

    "X7": "Repayment Status - 2 Months Ago",

    "X8": "Repayment Status - 3 Months Ago",

    "X9": "Repayment Status - 4 Months Ago",

    "X10": "Repayment Status - 5 Months Ago",

    "X11": "Repayment Status - 6 Months Ago",

    "X12": "Bill Amount - Previous Month (NT dollars)",

    "X13": "Bill Amount - 2 Months Ago (NT dollars)",

    "X14": "Bill Amount - 3 Months Ago (NT dollars)",

    "X15": "Bill Amount - 4 Months Ago (NT dollars)",

    "X16": "Bill Amount - 5 Months Ago (NT dollars)",

    "X17": "Bill Amount - 6 Months Ago (NT dollars)",

    "X18": "Payment Amount - Previous Month (NT dollars)",

    "X19": "Payment Amount - 2 Months Ago (NT dollars)",

    "X20": "Payment Amount - 3 Months Ago (NT dollars)",

    "X21": "Payment Amount - 4 Months Ago (NT dollars)",

    "X22": "Payment Amount - 5 Months Ago (NT dollars)",

    "X23": "Payment Amount - 6 Months Ago (NT dollars)",

}


# =========================================================
# 6. DYNAMIC INPUT DEFAULT VALUES
# =========================================================

defaults = {}

for feature in features:

    if feature in categorical_features:

        # Find the most common category in the dataset
        mode_values = dataset[feature].dropna().mode()

        if not mode_values.empty:

            defaults[feature] = str(
                mode_values.iloc[0]
            )

        else:

            st.error(
                f"No valid category found for {feature}."
            )

            st.stop()

    else:
        defaults[feature] = 0


# =========================================================
# 7. DYNAMIC CATEGORY OPTIONS
# =========================================================

category_options = {}

for feature in categorical_features:

    unique_values = (
        dataset[feature]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not unique_values:

        st.error(
            f"No category options available for {feature}."
        )

        st.stop()

    # Sort numeric categories numerically
    try:

        unique_values = sorted(
            unique_values,
            key=float
        )

    except ValueError:

        unique_values = sorted(unique_values)

    category_options[feature] = unique_values


# =========================================================
# 8. CATEGORY DESCRIPTIONS
# =========================================================

category_labels = {

    "X2": {
        "1": "Male",
        "2": "Female",
    },

    "X3": {
        "0": "Unknown",
        "1": "Graduate School",
        "2": "University",
        "3": "High School",
        "4": "Others",
        "5": "Unknown",
        "6": "Unknown",
    },

    "X4": {
        "0": "Unknown",
        "1": "Married",
        "2": "Single",
        "3": "Others",
    },

}


def format_category(feature):

    def formatter(value):

        return category_labels.get(
            feature, {}
        ).get(value, value)

    return formatter


def render_feature(feature):
    label = display_names.get(feature, feature)
    default = defaults[feature]

    if feature in categorical_features:
        options = category_options[feature]
        default_str = str(default)
        index = options.index(default_str) if default_str in options else 0
        return st.selectbox(
            label,
            options,
            index=index,
            format_func=format_category(feature),
            key=f"input_v3_{feature}",
        )

    if feature == "X5":
        numeric_values = pd.to_numeric(
            dataset[feature], errors="coerce"
        ).dropna()
        return st.number_input(
            label,
            min_value=0,
            max_value=int(numeric_values.max()),
            value=int(default),
            step=1,
            key=f"input_v3_{feature}",
        )

    if feature == "X1":
        numeric_values = pd.to_numeric(
            dataset[feature], errors="coerce"
        ).dropna()
        return st.number_input(
            label,
            min_value=0.0,
            max_value=float(numeric_values.max()),
            value=float(default),
            step=5000.0,
            key=f"input_v3_{feature}",
        )

    return st.number_input(
        label,
        value=float(default),
        step=1000.0,
        key=f"input_v3_{feature}",
    )


# =========================================================
# 9. CUSTOMER INPUT FORM
# =========================================================

input_groups = [
    (
        "Customer profile",
        "Credit limit and basic demographic information",
        ["X1", "X2", "X3", "X4", "X5"],
    ),
    (
        "Repayment history",
        "Repayment status recorded over the previous six months",
        ["X6", "X7", "X8", "X9", "X10", "X11"],
    ),
    (
        "Monthly bill amounts",
        "Outstanding bill amounts over the previous six months",
        ["X12", "X13", "X14", "X15", "X16", "X17"],
    ),
    (
        "Monthly payment amounts",
        "Payments made over the previous six months",
        ["X18", "X19", "X20", "X21", "X22", "X23"],
    ),
]
grouped_features = {feature for _, _, group in input_groups for feature in group}
remaining_features = [feature for feature in features if feature not in grouped_features]
if remaining_features:
    input_groups.append(("Additional information", "", remaining_features))

with st.form("prediction_form", border=False):
    values = {}

    for group_title, group_note, group_features in input_groups:
        st.markdown(
            f'<div class="section-heading">{group_title}</div>',
            unsafe_allow_html=True,
        )
        if group_note:
            st.markdown(
                f'<div class="section-note">{group_note}</div>',
                unsafe_allow_html=True,
            )
        cols = st.columns(3)
        for index, feature in enumerate(group_features):
            with cols[index % 3]:
                values[feature] = render_feature(feature)

    st.markdown("")
    submitted = st.form_submit_button(
        "Estimate default risk",
        type="primary",
        use_container_width=True,
    )


# =========================================================
# 10. PREDICT CUSTOMER DEFAULT RISK
# =========================================================

if submitted:

    try:

        input_df = pd.DataFrame(
            [values],
            columns=features
        )

        # Convert categorical features to strings
        for column in categorical_features:

            input_df[column] = (
                input_df[column].astype(str)
            )

        # Get model class labels
        model_classes = list(model.classes_)

        if 1 not in model_classes:

            st.error(
                "The trained model does not contain "
                "class 1 (Default). Check your training target."
            )

            st.stop()

        # Find the probability column for class 1
        default_class_index = model_classes.index(1)

        # Predict probabilities
        probabilities = model.predict_proba(input_df)[0]

        probability = float(
            probabilities[default_class_index]
        )

        # Probability of no default
        no_default_probability = 1 - probability

        # Classification threshold
        prediction = int(probability >= 0.5)

        # ---------------------------------------------
        # DISPLAY RESULTS
        # ---------------------------------------------

        st.divider()

        with st.container(border=True):

            st.subheader("Customer Prediction Result")
            st.caption("Estimated probabilities from the trained model")

            left, right = st.columns(2)

            with left:
                with st.container(border=True):
                    st.metric(
                        "Default Probability",
                        f"{probability:.1%}"
                    )

            with right:
                with st.container(border=True):
                    st.metric(
                        "No-Default Probability",
                        f"{no_default_probability:.1%}"
                    )

            st.progress(probability)
            st.caption("Probability of default · 0% to 100%")

            if prediction == 1:
                st.error(
                    "Model classification: Default",
                    icon="⚠️"
                )
            else:
                st.success(
                    "Model classification: No Default",
                    icon="✅"
                )

            st.caption(
                "Classification uses a demonstration threshold of 0.50."
            )

        st.warning(
            "Educational model output only—not a reliable prediction "
            "about an individual or a recommendation to approve or "
            "reject credit."
        )

    except Exception as exc:

        st.error(
            f"Prediction error: {exc}"
        )


# =========================================================
# 11. HISTORICAL DATA EXPLORER
# =========================================================

st.markdown(
    '<div class="section-heading">Explore historical records</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="section-note">Filter anonymized records from the public historical dataset.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.caption(
        "These records contain no customer names or IDs. They represent historical "
        "dataset examples, not identifiable or current customers."
    )

    age_values = pd.to_numeric(dataset["X5"], errors="coerce").dropna()
    age_min = int(age_values.min())
    age_max = int(age_values.max())
    gender_values = category_options.get("X2", [])
    target_column = info.get("target", "default.payment.next.month")

    filter_cols = st.columns(4)
    with filter_cols[0]:
        age_range = st.slider(
            "Customer age",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max),
            key="record_age_filter",
        )
    with filter_cols[1]:
        gender_filter = st.selectbox(
            "Gender",
            ["Any"] + gender_values,
            format_func=lambda value: (
                "Any gender"
                if value == "Any"
                else category_labels.get("X2", {}).get(value, value)
            ),
            key="record_gender_filter",
        )
    with filter_cols[2]:
        outcome_filter = st.selectbox(
            "Default next month",
            ["Any outcome", "No default", "Default"],
            key="record_outcome_filter",
        )
    with filter_cols[3]:
        row_limit = st.selectbox(
            "Rows to display",
            [10, 25, 50, 100, 250],
            index=1,
            key="record_row_limit",
        )

    filtered_records = dataset.copy()
    filtered_records["X5"] = pd.to_numeric(
        filtered_records["X5"], errors="coerce"
    )
    filtered_records = filtered_records[
        filtered_records["X5"].between(age_range[0], age_range[1])
    ]

    if gender_filter != "Any":
        filtered_records = filtered_records[
            filtered_records["X2"].astype(str) == gender_filter
        ]

    if outcome_filter != "Any outcome" and target_column in filtered_records.columns:
        target_values = pd.to_numeric(
            filtered_records[target_column], errors="coerce"
        )
        outcome_value = 1 if outcome_filter == "Default" else 0
        filtered_records = filtered_records[target_values == outcome_value]

    count_col, description_col = st.columns([1, 3])
    count_col.metric("Matching records", f"{len(filtered_records):,}")
    description_col.caption(
        f"Showing up to {row_limit} matching rows. Use the table toolbar to search "
        "or download this filtered view."
    )

    visible_records = filtered_records.head(row_limit).copy()
    visible_records = visible_records.rename(
        columns={
            **display_names,
            target_column: "Default next month",
        }
    )
    if "Default next month" in visible_records.columns:
        visible_records["Default next month"] = (
            visible_records["Default next month"]
            .map({0: "No", 1: "Yes"})
            .fillna("Unknown")
        )

    st.dataframe(
        visible_records,
        use_container_width=True,
        hide_index=True,
        height=440,
    )


# =========================================================
# 12. MODEL INSIGHTS
# =========================================================

st.markdown('<div class="section-heading">Model insights</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Performance and feature information for context—not a guarantee of future results.</div>',
    unsafe_allow_html=True,
)

metrics = info.get("test_metrics", {})
valid_metrics = {}
for name, value in metrics.items():
    try:
        valid_metrics[name] = float(value)
    except (ValueError, TypeError):
        continue

if valid_metrics:
    with st.container(border=True):
        st.markdown("#### Held-out test performance")
        metric_cols = st.columns(min(5, len(valid_metrics)))
        for col, (name, value) in zip(metric_cols, valid_metrics.items()):
            col.metric(name.replace("_", " ").title(), f"{value:.3f}")
        chart_df = pd.DataFrame({
            "Metric": list(valid_metrics.keys()),
            "Score": list(valid_metrics.values()),
        })
        st.bar_chart(chart_df.set_index("Metric"), height=240)

importance_path = DATA_DIR / "feature_importance.csv"
comparison_path = DATA_DIR / "model_comparison.csv"
importance_col, comparison_col = st.columns(2)

with importance_col:
    with st.container(border=True):
        st.markdown("#### Important model features")
        if importance_path.exists():
            try:
                importance_df = pd.read_csv(importance_path)
                if {"Feature", "Importance"}.issubset(importance_df.columns):
                    importance_df["Importance"] = pd.to_numeric(
                        importance_df["Importance"], errors="coerce"
                    )
                    importance_df = importance_df.dropna(subset=["Importance"])
                    raw_features = importance_df["Feature"].astype(str)
                    importance_df["Feature"] = (
                        raw_features.map(display_names).fillna(raw_features)
                    )
                    top_features = importance_df.sort_values(
                        "Importance", ascending=False
                    ).head(10)
                    if not top_features.empty:
                        st.bar_chart(
                            top_features.set_index("Feature")["Importance"],
                            horizontal=True,
                            sort="ascending",
                            height=330,
                        )
                    else:
                        st.info("No valid feature importance values found.")
                else:
                    st.warning("The feature importance file has an unexpected format.")
            except Exception as exc:
                st.error(f"Error reading feature importance: {exc}")
        else:
            st.info("Run the model evaluation notebook to generate feature importance.")

with comparison_col:
    with st.container(border=True):
        st.markdown("#### Model comparison")
        if comparison_path.exists():
            try:
                comparison_df = pd.read_csv(comparison_path)
                required_columns = [
                    "Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"
                ]
                if all(column in comparison_df.columns for column in required_columns):
                    st.dataframe(
                        comparison_df,
                        use_container_width=True,
                        hide_index=True,
                    )
                    comparison_chart = comparison_df.set_index("Model")[
                        ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
                    ]
                    st.bar_chart(comparison_chart, height=280)
                else:
                    st.warning("The model comparison file is missing expected columns.")
            except Exception as exc:
                st.error(f"Error reading model comparison: {exc}")
        else:
            st.info("Run the model evaluation notebook to generate model comparison.")


# =========================================================
# 13. ABOUT AND DATASET
# =========================================================

with st.container(border=True):
    st.markdown("#### About this model")
    about_cols = st.columns(3)
    about_cols[0].caption("MODEL")
    about_cols[0].write(info.get("model_name", "Not available"))
    about_cols[1].caption("DATASET")
    about_cols[1].write(info.get("dataset", "Not available"))
    about_cols[2].caption("TARGET")
    about_cols[2].write(
        f"{info.get('target', 'Not available')} · 1 means default next month"
    )
    st.caption(
        "Limitations: Historical Taiwan data from 2005 may not represent current "
        "borrowers or other countries. The model may contain bias and must not be "
        "used for real lending decisions."
    )

st.markdown(
    """
    <div class="app-footer">
      UCI Default of Credit Card Clients · Yeh (2009) · DOI: 10.24432/C55S3H · CC BY 4.0<br>
      For educational and research purposes only
    </div>
    """,
    unsafe_allow_html=True,
)
