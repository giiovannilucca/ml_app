from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR

def get_classifiers() -> dict:
    """
    Returns a dictionary of classifier models.

    Returns:
    - Dictionary with model name as key and instantiated classifier as value.
    """
    return {
        "Logistic Regression": LogisticRegression(solver='liblinear'),
        "KNN": KNeighborsClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Support Vector Machine": SVC()
    }

def get_regressors() -> dict:
    """
    Returns a dictionary of regressor models.

    Returns:
    - Dictionary with model name as key and instantiated regressor as value.
    """
    return {
        "Linear Regression": LinearRegression(),
        "KNN": KNeighborsRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Support Vector Machine": SVR()
    }