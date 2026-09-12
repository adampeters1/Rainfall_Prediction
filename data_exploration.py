import pandas as pd
pd.set_option('display.max_rows', 10000)
pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 200)

from baseline_model import model_trainer

def inspect(df):
    print(df.head(1))
    print(df.info())
    print(df.describe())
    print(df.rain_event.nunique())
    print(df.rain_event.unique())
    print(df.rain_event.value_counts())


def main():
    df = pd.read_parquet("./Data/train.parquet")
    test_df = pd.read_parquet("./Data/test.parquet")
    val_df = pd.read_parquet("./Data/validation.parquet")

    #inspect(df)
    #inspect(test_df)
    #inspect(val_df)

    print(df.shape)
    print(test_df.shape)
    print(val_df.shape)

    # Check for events that should have been classified under standard definition of rainfall > 0.1mm:
    bad_rows = df[(df.rain_event == 0) & (df.rain_rate_mm_per_hr >= 0.01)][["rain_rate_mm_per_hr", "rain_event"]]
    print(f"Example:\n{bad_rows.head(1)}\nNumber of entries that are incorrect: {len(bad_rows)} out of {df.shape[0]}.")

    # Run test model baseline (Will be moved to pipeline later)
    obj = model_trainer()
    print(obj.train())


if __name__ == "__main__":
    main()