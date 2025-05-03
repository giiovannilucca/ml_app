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

# Step 1 - Upload CSV
st.subheader("Step 1 - Select database")
uploaded_file = st.file_uploader("Database in .csv format", type="csv")

df = None
if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
        st.dataframe(df)
    except ValueError as e:
        st.error(str(e))

# Step 2 - Select ML task
st.subheader("Step 2 - Select ML task")
problem_type = st.radio("ML task:", options=["Classification", "Regression"])

# Step 3 - Select target column
st.subheader("Step 3 - Select target")
if df is not None:
    target_column = st.selectbox("Target column:", df.columns)
    if problem_type == "Classification" and target_column:
        unique_classes = sorted(df[target_column].dropna().unique())
        st.markdown("### Rename class labels (Optional)")
        st.markdown("You can provide more intuitive names for each class below.")

        rename_map = {}
        index_map = {}
        description_map = {}

        with st.form("class_label_editor"):
            for original_class in unique_classes:
                col1, col2 = st.columns([1, 3])
                with col1:
                    idx = st.selectbox(f"Index for `{original_class}`", list(range(len(unique_classes))), key=f"idx_{original_class}")
                    index_map[original_class] = idx
                with col2:
                    desc = st.text_input(f"Description for `{original_class}`", key=f"desc_{original_class}")
                    description_map[original_class] = desc

            submitted = st.form_submit_button("Apply changes")
        if submitted:
            st.success("Class mappings updated successfully.")
            df[target_column] = df[target_column].replace(index_map)
else:
    target_column = st.selectbox("Target column:", options=["None"])

# Step 4 - Select features
st.subheader("Step 4 - Remove attributes (optional)")
if df is not None and target_column:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_columns = [col for col in numeric_cols if col != target_column]
    selected_features = st.multiselect("Attribute columns:", feature_columns, default=feature_columns)
else:
    selected_features = st.multiselect("Attribute columns:", ["None"], default=["None"])

# Step 5 - Train/test split
st.subheader("Step 5 - Select training subset percentage")
train_size = st.slider("Training subset size (%):", min_value=10, max_value=90, value=70, step=5)

if df is not None and target_column and selected_features:
    try:
        X_train, X_test, y_train, y_test = split_data(df, target_column, selected_features, train_size)
        st.success(
            f"**Training subset:** {X_train.shape[0]} instances\n\n"
            f"**Test subset:** {X_test.shape[0]} instances"
        )

        if problem_type == "Classification":
            inverse_index_map = {v: description_map[k] for k, v in index_map.items()}

            train_counts = pd.Series(y_train).value_counts().to_dict()
            test_counts = pd.Series(y_test).value_counts().to_dict()
            
            details = "#### 🔍 Class distribution per subset\n"
            details += "**Training subset:**\n"
            for label, count in train_counts.items():
                if len(inverse_index_map) == 1:
                    class_name = str(label)                  
                else:
                    class_name = inverse_index_map.get(label, str(label))
                details += f"- Class **{class_name}**: {count} instances\n"

            details += "\n\n**Test subset:**\n"            
            for label, count in test_counts.items():
                if len(inverse_index_map) == 1:
                    class_name = str(label)                  
                else:
                    class_name = inverse_index_map.get(label, str(label))
                details += f"- Class **{class_name}**: {count} instances\n"

            st.markdown(details)

    except Exception as e:
        st.error(f"Error during train/test split: {e}")

# Step 6 - ML algorithms
st.subheader("Step 6 - Select ML algorithm(s)")
model_options = ["None"]
if df is not None and problem_type:
    if problem_type == "Classification":
        model_options += list(get_classifiers().keys()) + ["All"]
    elif problem_type == "Regression":
        model_options += list(get_regressors().keys()) + ["All"]

model_choice = st.selectbox("Select a model:", options=model_options, index=0)

# Step 7 - Train and evaluate
st.subheader("Step 7 - Start training and evaluation")
if st.button("Start"):
    if df is None or target_column is None or not selected_features:
        st.warning("Please complete all steps before training.")
    elif model_choice == "None":
        st.warning("Please choose a model.")
    else:
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
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, hide_index=True)

st.markdown(
    """
    <hr style="margin-top: 40px; margin-bottom: 10px;"/>

    <div style="text-align: center; font-size: 0.85em; color: gray;">
        © 2025 Data Inception. For educational and research purposes. All rights reserved.
    </div>
    """,
    unsafe_allow_html=True
)