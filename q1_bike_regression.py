import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def design_matrix_linear(X):
    """Add intercept column."""
    return np.hstack([np.ones((X.shape[0], 1)), X])

def normal_equation(X, y):
    """
    Solve normal equation using pseudoinverse:
    theta = (X^T X)^+ X^T y
    Avoids singular matrix error.
    """
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    Xty = X.T @ y
    theta = XtX_inv @ Xty
    return theta

def poly_features_no_interaction(X, degree):
    cols = []
    for d in range(1, degree + 1):
        cols.append(X ** d)
    return np.hstack(cols)

def quadratic_with_interactions(X):
    n, p = X.shape
    quad_terms = []
    for i in range(p):
        for j in range(i, p):
            quad_terms.append((X[:, i] * X[:, j]).reshape(-1, 1))
    quad_terms = np.hstack(quad_terms)
    return np.hstack([X, quad_terms])

def load_and_preprocess(path):
    df = pd.read_csv(path, parse_dates=['datetime'])

    for col in ['casual', 'registered']:
        if col in df.columns:
            df = df.drop(columns=[col])

    df['hour'] = df['datetime'].dt.hour
    df['day'] = df['datetime'].dt.day
    df['month'] = df['datetime'].dt.month
    df['year'] = df['datetime'].dt.year
    df['weekday'] = df['datetime'].dt.weekday

    numeric_cols = ['temp', 'atemp', 'humidity', 'windspeed', 'year', 'day']
    categorical_cols = ['season', 'holiday', 'workingday', 'weather',
                        'hour', 'month', 'weekday']

    df = df[numeric_cols + categorical_cols + ['count']]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('ohe', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer([
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    X_train = preprocessor.fit_transform(train_df[numeric_cols + categorical_cols])
    X_test = preprocessor.transform(test_df[numeric_cols + categorical_cols])

    y_train = train_df['count'].values.reshape(-1, 1)
    y_test = test_df['count'].values.reshape(-1, 1)

    return X_train, X_test, y_train, y_test

def evaluate(train_csv):
    X_train, X_test, y_train, y_test = load_and_preprocess(train_csv)
    results = []

    # Linear Regression
    Xd_train = design_matrix_linear(X_train)
    theta = normal_equation(Xd_train, y_train)
    y_pred = design_matrix_linear(X_test) @ theta
    results.append(("Linear Regression", mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)))

    # Polynomial models
    for d in [2, 3, 4]:
        Xp_train = poly_features_no_interaction(X_train, d)
        Xp_test = poly_features_no_interaction(X_test, d)
        theta = normal_equation(design_matrix_linear(Xp_train), y_train)
        y_pred = design_matrix_linear(Xp_test) @ theta
        results.append((f"Polynomial Degree {d}", mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)))

    # Quadratic with interactions
    Xq_train = quadratic_with_interactions(X_train)
    Xq_test = quadratic_with_interactions(X_test)
    theta = normal_equation(design_matrix_linear(Xq_train), y_train)
    y_pred = design_matrix_linear(Xq_test) @ theta
    results.append(("Quadratic with Interactions", mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)))

    print("\n=== Test Set Results ===\n")
    for model, mse, r2 in results:
        print(f"{model:30s} | MSE = {mse:.4f} | R2 = {r2:.4f}")


if __name__ == "__main__":
    evaluate("train.csv")
