import pandas as pd

from sklearn.model_selection import train_test_split
from typing import Tuple

def split_data(df: pd.DataFrame, target_column: str, selected_features: list, train_size: float) -> Tuple:
    """
    Splits the dataset into training and testing sets based on the provided features and target column.

    Parameters:
    - df: pandas.DataFrame
        The input dataset containing the features and target column.
    - target_column: str
        The name of the target column in the dataset to be predicted.
    - selected_features: list of str
        A list of feature column names to be used for model training.
    - train_size: float
        The proportion of data to be used for training (between 0.0 and 1.0).

    Returns:
    - Tuple
        A tuple containing:
        - X_train: pandas.DataFrame
            The training set features.
        - X_test: pandas.DataFrame
            The testing set features.
        - y_train: pandas.Series
            The training set target values.
        - y_test: pandas.Series
            The testing set target values.

    Example:
    --------
    X_train, X_test, y_train, y_test = split_data(df, 'target_column', ['feature1', 'feature2'], 0.7)
    """
    X = df[selected_features]
    y = df[target_column]
    return train_test_split(X, y, train_size=train_size / 100, random_state=42)

