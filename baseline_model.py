import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, root_mean_squared_error


class model_trainer():
    def __init__(self) -> None:
        pass

    def train(self):
        df = pd.read_parquet("./Data/train.parquet")
        test_df = pd.read_parquet("./Data/test.parquet")
        val_df = pd.read_parquet("./Data/validation.parquet")

        if "timestamp" in df.columns:
            df.drop(columns=["timestamp", "rain_event", "simulation_id"], inplace=True)

        if "timestamp" in test_df.columns:
            test_df.drop(columns=["timestamp",  "rain_event", "simulation_id"], inplace=True)

        TARGET="rain_rate_mm_per_hr"

        X_train=df.drop(columns=[TARGET]).copy()
        y_train=df[TARGET].copy()

        X_test=test_df.drop(columns=[TARGET]).copy()
        y_test=test_df[TARGET].copy()

        numerical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )


        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]
        )


        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numerical",
                    numerical_pipeline,
                    make_column_selector(dtype_include="number")
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    make_column_selector(dtype_include=["object", "category"])
                )
            ]
        )

        model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression())
        ]
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)

        result = f"RMSE for LR: {rmse:.3f}"
        
        print(f"R2 Score: {r2:.3f}")
        print(f"MAE: {mae:.3f}")
        print(f"MSE: {mse:.3f}")
        print(f"RMSE: {rmse:.3f}")

        return result


def main():
    obj = model_trainer()
    print(obj.train())


if __name__ == "__main__":
    main()