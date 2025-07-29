import streamlit as st
import pandas as pd
import numpy as np

from utils.data_loader import load_csv, load_image_base64
from utils.preprocessing import split_data
from utils.models import get_classifiers, get_regressors
from utils.evaluation import evaluate_classification_model, evaluate_regression_model

st.set_page_config(layout="wide")

encoded_logo = load_image_base64("etc/logo.png")

st.markdown(
    f"""
    <div style="text-align: left;">
        <img src="data:image/png;base64,{encoded_logo}" alt="Logo" style="width:600px;"><br>
        <p style="font-size:18px;"> This portal enables the training and evaluation of predictive models in a simple and guided way.</p>
    </div>
    """,
    unsafe_allow_html=True
)

if 'df' not in st.session_state:
    st.session_state.df = None

if "target_column" not in st.session_state:
    st.session_state.target_column = None

# if "index_map" not in st.session_state:
#     st.session_state.index_map = {}

# if "description_map" not in st.session_state:
#     st.session_state.description_map = {}

# Step 1 - Upload CSV
st.subheader("Step 1 - Select database")
uploaded_file = st.file_uploader("Database in .csv format:", type="csv")

if uploaded_file is None:
    st.session_state.df = None
else:
    df = load_csv(uploaded_file)
    # st.dataframe(st.session_state.df)
    st.session_state.df = df

# Step 2 - Select ML task
st.subheader("Step 2 - Select ML task")
problem_type = st.radio("ML task:", options=["Classification", "Regression"])

# Step 3 - Select target column
st.subheader("Step 3 - Select target")
if "df" in st.session_state and st.session_state.df is not None:
    target_column = st.selectbox("Target column:", st.session_state.df.columns)
    st.session_state.target_column = target_column
else:
    target_column = st.selectbox("Target column:", options=["None"])

# Step 4 - Select features
st.subheader("Step 4 - Remove attributes (optional)")
if "df" in st.session_state and st.session_state.df is not None and target_column:
    numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in numeric_cols if col != target_column]
    selected_features = st.multiselect("Attribute columns:", feature_columns, default=feature_columns)
else:
    selected_features = st.multiselect("Attribute columns:", ["None"], default=["None"])

# Step 5 - Train/test split
st.subheader("Step 5 - Select training subset percentage")
train_size = st.slider("Training subset size (%):", min_value=10, max_value=90, value=70, step=5)

if "df" in st.session_state and st.session_state.df is not None and st.session_state.target_column and selected_features:
    try:
        X_train, X_test, y_train, y_test = split_data(st.session_state.df, st.session_state.target_column, selected_features, train_size)
        # st.success(
        #     f"**Training subset:** {X_train.shape[0]} instances with {X_train.shape[1]} attributes\n\n"
        #     f"**Test subset:** {X_test.shape[0]} instances with {X_test.shape[1]} attributes"
        # )

        if problem_type == "Classification":
            train_counts = pd.Series(y_train).value_counts().to_dict()
            test_counts = pd.Series(y_test).value_counts().to_dict()
            
            details = f"**Training subset:** {X_train.shape[0]} instances with {X_train.shape[1]} attribute(s)\n"
            for label, count in train_counts.items():
                details += f"- Class **{label}**: {count} instances\n"

            details += f"\n\n**Test subset:** {X_test.shape[0]} instances with {X_test.shape[1]} attribute(s)\n"            
            for label, count in test_counts.items():
                details += f"- Class **{label}**: {count} instances\n"

            st.markdown(details)

    except Exception as e:
        st.error(f"Error during train/test split: {e}")

# Step 6 - ML algorithms
st.subheader("Step 6 - Select ML algorithm(s)")
model_options = ["None"]
if "df" in st.session_state and st.session_state.df is not None and problem_type:
    if problem_type == "Classification":
        model_options += list(get_classifiers().keys()) + ["All"]
    elif problem_type == "Regression":
        model_options += list(get_regressors().keys()) + ["All"]

model_choice = st.selectbox("Select a model:", options=model_options, index=0)

# Step 7 - Train and evaluate
st.subheader("Step 7 - Start training and evaluation")
if st.button("Start"):
    if "df" not in st.session_state or st.session_state.df is None or st.session_state.target_column is None or not selected_features:
        st.warning("Please complete all steps before training.")
    elif model_choice == "None":
        st.warning("Please choose a model.")
    else:
        try:
            X_train, X_test, y_train, y_test = split_data(st.session_state.df, st.session_state.target_column, selected_features, train_size)
        except Exception as e:
            st.error(f"Error during train/test split: {e}")
            st.stop()

        results = []
        if problem_type == "Classification":
            classifiers = get_classifiers()
            models = classifiers.items() if model_choice == "All" else [(model_choice, classifiers[model_choice])]
            for name, model in models:
                metrics = evaluate_classification_model(name, model, X_train, X_test, y_train, y_test)
                results.append(metrics)

        elif problem_type == "Regression":
            regressors = get_regressors()
            models = regressors.items() if model_choice == "All" else [(model_choice, regressors[model_choice])]
            for name, model in models:
                metrics = evaluate_regression_model(name, model, X_train, X_test, y_train, y_test)
                results.append(metrics)

        if results:
            st.write("Results obtained in the test subset")

            if problem_type == "Classification":
                unique_labels = np.unique(y_test)
                if len(unique_labels) == 2:
                    pos_label = unique_labels[1]
                    st.write(f"For binary classification, the positive class was assumed to be: **{pos_label}**")

            results_df = pd.DataFrame(results)
            st.dataframe(results_df, hide_index=True)

# Rodapé
st.markdown(
    """
    <hr style="margin-top: 40px; margin-bottom: 10px;"/>

    <div style="text-align: center; font-size: 0.85em; color: gray;">
        © 2025 Data Inception. For educational and research purposes. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)