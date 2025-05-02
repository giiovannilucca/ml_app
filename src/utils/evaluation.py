import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt

from typing import List, Dict
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    balanced_accuracy_score, r2_score, mean_squared_error, mean_absolute_error
)

def evaluate_classification_model(name: str, model, X_train, X_test, y_train, y_test) -> Dict:
    """
    Evaluates a classification model.

    Parameters:
    - name: Name of the model.
    - model: Instantiated classifier.
    - X_train, X_test: Training and test features.
    - y_train, y_test: Training and test labels.

    Returns:
    - Dictionary with evaluation metrics.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {"Model": name, "Accuracy": f"{accuracy_score(y_test, y_pred) * 100:.2f}%"}

    if len(np.unique(y_test)) == 2:
        metrics["Precision"] = f"{precision_score(y_test, y_pred, zero_division=0) * 100:.2f}%"
        metrics["Recall"] = f"{recall_score(y_test, y_pred, zero_division=0) * 100:.2f}%"
        metrics["F1 Score"] = f"{f1_score(y_test, y_pred, zero_division=0) * 100:.2f}%"
    else:
        metrics["Balanced Accuracy"] = f"{balanced_accuracy_score(y_test, y_pred) * 100:.2f}%"

    # cm = confusion_matrix(y_test, y_pred)
    # fig, ax = plt.subplots()
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    # ax.set_title(f"Confusion Matrix - {name}")
    # ax.set_xlabel("Predicted")
    # ax.set_ylabel("Actual")
    # st.pyplot(fig)

    return metrics

def evaluate_regression_model(name: str, model, X_train, X_test, y_train, y_test) -> Dict:
    """
    Evaluates a regression model.

    Parameters:
    - name: Name of the model.
    - model: Instantiated regressor.
    - X_train, X_test: Training and test features.
    - y_train, y_test: Training and test targets.

    Returns:
    - Dictionary with evaluation metrics.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return {
        "Model": name,
        "R²": r2_score(y_test, y_pred),
        "Mean Squared Error (MSE)": mean_squared_error(y_test, y_pred),
        "Mean Absolute Error (MAE)": mean_absolute_error(y_test, y_pred)
    }