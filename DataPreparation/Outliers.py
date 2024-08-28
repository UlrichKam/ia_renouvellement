### When dealing with outliers, you can:
# Trim: remove the outliers
# Cap: Set outliers to a min/max value 
# Cosider Missing: Treat it as a missing value


### Identification of outliers:
# For normal distributions The Empirical Rule of Normal distribution apply, outliers are values below [mean - 3(std)] or above [mean + 3(std)]

# For Skewed distributions, Inter-Quartile Range proximity rule, outliers are values below [Q1 - (1.5IQR)] or above [Q3 + (1.5IQR)]

# For other distributions, use the percentile method, outliers are values below the 1% percentile/quantile or above the 99% percentile/quantile


import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import pandas as pd
import mlflow
from globals import TMP_DIRECTORY

warnings.filterwarnings('ignore')


class Outliers:
    def visualise_outliers(df: pd.DataFrame, columns: list[str], temp_path: str = TMP_DIRECTORY,
                           log_artefacts: bool = True, log_path_argument: str = ""):
        """
        Creates visualisations of the missing values in the dataframe provided as a parameter
        """
        # Box plot
        if columns is None:
            columns = df.columns

        for column in columns:
            plt.figure()
            diagram = sns.boxplot(x=df[column])
            diagram.figure.savefig(temp_path + f"/outlier box ({column}).png")
            if log_artefacts:
                mlflow.log_artifact(temp_path + f"/outlier box ({column}).png", artifact_path=f"Outlier{log_path_argument}")

            plt.figure()
            diagram = sns.distplot(x=df[column])
            diagram.figure.savefig(temp_path + f"/outlier dist ({column}).png")
            if log_artefacts:
                mlflow.log_artifact(temp_path + f"/outlier dist ({column}).png",
                                    artifact_path=f"Outlier{log_path_argument}")


    def trim_outliers(df: pd.DataFrame, columns: list[str], upper_bound: float = 0.99, lower_bound: float = 0.01, log_metric: bool = True):
        if columns is None:
            columns = df.columns

        for column in columns:
            if column not in df.columns:
                print(f"Column {column} not found in dataframe.")

            elif df[column].dtype != int and df[column].dtype != float:
                print(f"Column {column} is not numeric.")

            else:
                upper_quantile = df[column].quantile(upper_bound)
                lower_quantile = df[column].quantile(lower_bound)
                upper_quantile_count = float(df[df[column] > upper_quantile].shape[0])
                lower_quantile_count = float(df[df[column] < lower_quantile].shape[0])

                df[column] = df[lower_quantile < df[column] < upper_quantile]

                if log_metric:
                    mlflow.log_metrics({
                        f"Outliers.Cap.{column}.UpperQuantile": upper_quantile,
                        f"Outliers.Cap.{column}.LowerQuantile": lower_quantile,
                        f"Outliers.Cap.{column}.UpperQuantileCount": upper_quantile_count,
                        f"Outliers.Cap.{column}.LowerQuantileCount": lower_quantile_count
                    })
        return df

    def cap_outliers(df: pd.DataFrame, columns: list[str], upper_bound: float = 0.99, lower_bound: float = 0.01, log_metric: bool = True):
        if columns is None:
            columns = df.columns

        for column in columns:
            if column not in df.columns:
                print(f"Column {column} not found in dataframe.")

            elif df[column].dtype != int and df[column].dtype != float:
                print(f"Column {column} is not numeric.")

            else:
                upper_quantile = df[column].quantile(upper_bound)
                lower_quantile = df[column].quantile(lower_bound)
                upper_quantile_count = float(df[df[column] > upper_quantile].shape[0])
                lower_quantile_count = float(df[df[column] < lower_quantile].shape[0])

                df[column] = df[column].apply(lambda x: x if x > lower_quantile else lower_quantile)
                df[column] = df[column].apply(lambda x: x if x < upper_quantile else upper_quantile)

                if log_metric:
                    mlflow.log_metrics({
                        f"Outliers.Cap.{column}.UpperQuantile": upper_quantile,
                        f"Outliers.Cap.{column}.LowerQuantile": lower_quantile,
                        f"Outliers.Cap.{column}.UpperQuantileCount": upper_quantile_count,
                        f"Outliers.Cap.{column}.LowerQuantileCount": lower_quantile_count
                    })
        return df

    def null_outliers(df: pd.DataFrame, columns: list[str], upper_bound: float = 0.99, lower_bound: float = 0.01, log_metric: bool = True):
        if columns is None:
            columns = df.columns

        for column in columns:
            if column not in df.columns:
                print(f"Column {column} not found in dataframe.")

            elif df[column].dtype != int and df[column].dtype != float:
                print(f"Column {column} is not numeric.")

            else:
                upper_quantile = df[column].quantile(upper_bound)
                lower_quantile = df[column].quantile(lower_bound)
                upper_quantile_count = float(df[df[column] > upper_quantile].shape[0])
                lower_quantile_count = float(df[df[column] < lower_quantile].shape[0])

                df[column] = df[column].apply(lambda x: x if x > lower_quantile else None)
                df[column] = df[column].apply(lambda x: x if x < upper_quantile else None)

                if log_metric:
                    mlflow.log_metric(f"Outliers.Null.{column}.UpperQuantile", upper_quantile)
                    mlflow.log_metric(f"Outliers.Null.{column}.LowerQuantile", lower_quantile)
                    mlflow.log_metric(f"Outliers.Null.{column}.UpperQuantileCount", upper_quantile_count)
                    mlflow.log_metric(f"Outliers.Null.{column}.LowerQuantileCount", lower_quantile_count)
        return df
