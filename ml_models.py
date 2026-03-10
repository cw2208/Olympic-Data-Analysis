import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def split_train_test_by_year(model_df, split_year=2000):
    """
    Split model data into train and test sets by year.
    """
    train_df = model_df[model_df["Year"] < split_year]
    test_df = model_df[model_df["Year"] >= split_year]
    return train_df, test_df


def train_models(model_df, split_year=2000):
    """
    Train multiple regression models and return fitted models and metrics.
    """
    features = ["GDP", "Population", "Previous_Medal_Count"]
    target = "Medal_Count"

    train_df, test_df = split_train_test_by_year(model_df, split_year)

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge alpha=0.1": Ridge(alpha=0.1),
        "Ridge alpha=1.0": Ridge(alpha=1.0),
        "Ridge alpha=10.0": Ridge(alpha=10.0),
        "Random Forest 100": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        ),
        "Random Forest 200": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        )
    }

    fitted_models = {}
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = mean_squared_error(y_test, predictions) ** 0.5
        r2 = r2_score(y_test, predictions)

        fitted_models[name] = model
        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

    results_df = pd.DataFrame(results).sort_values(by="RMSE")
    return fitted_models, results_df


def plot_model_rmse(results_df):
    """
    Plot RMSE values for regression models.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="Model", y="RMSE")
    plt.title("Regression Model Comparison by RMSE")
    plt.xlabel("Model")
    plt.ylabel("RMSE")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig("results/model_rmse_comparison.png", dpi=300,
                bbox_inches="tight")
    plt.close()


def save_predictions(model_df, model, split_year=2000):
    """
    Save actual and predicted medal counts for the test set.
    """
    features = ["GDP", "Population", "Previous_Medal_Count"]
    target = "Medal_Count"

    _, test_df = split_train_test_by_year(model_df, split_year)
    X_test = test_df[features]
    y_test = test_df[target]

    predictions = model.predict(X_test)

    output_df = test_df[["Year", "NOC"]].copy()
    output_df["Actual_Medal_Count"] = y_test.values
    output_df["Predicted_Medal_Count"] = predictions
    output_df.to_csv("results/model_predictions.csv", index=False)