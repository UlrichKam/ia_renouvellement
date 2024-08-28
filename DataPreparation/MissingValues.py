import mlflow
import pandas as pd
import numpy as np
import datetime as dt
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from globals import TMP_DIRECTORY


class MissingValues:
    """
    A class that deals with missing values in a Dataframe via it's methods
    """

    """
    def __init__(self, df:pd.DataFrame):
        self.df = df
    """

    def describe_missing(df: pd.DataFrame, log_metrics: bool = True) -> None:
        # TODO: Finish this function
        # Describe per column
        #
        mv_df = df.isna().sum().rename_axis('column_names').reset_index(name='no_missing')
        list_of_columns_with_missing_values = mv_df[mv_df["no_missing"] != 0]["column_names"].tolist()
        list_of_columns_without_missing_values = mv_df[mv_df["no_missing"] == 0]["column_names"].tolist()

    def visualise_missing(df: pd.DataFrame, temp_path: str = TMP_DIRECTORY, log_artefacts: bool = True,
                          log_path_argument: str = ""):
        """
        Creates visualisations of the missing values in the dataframe provided as a parameter
        """
        # Bar plot
        msno.bar(df)
        plt.savefig(temp_path + "/missing bar.png")
        if log_artefacts:
            mlflow.log_artifact(temp_path + "/missing bar.png", artifact_path=f"Missing{log_path_argument}")

        # Matrix plot
        msno.matrix(df)
        plt.savefig(temp_path + "/missing matrix.png")
        if log_artefacts:
            mlflow.log_artifact(temp_path + "/missing matrix.png", artifact_path=f"Missing{log_path_argument}")

        # dendrogram plot
        msno.dendrogram(df)
        plt.savefig(temp_path + "/missing dendrogram.png")
        # msno.matrix(df).get_figure().savefig(temp_path + "/missing dendrogram.png")
        if log_artefacts:
            mlflow.log_artifact(temp_path + "/missing dendrogram.png", artifact_path=f"Missing{log_path_argument}")

    def trim_by_percentage(df: pd.DataFrame, percentage_null_values: float, columns: list[str],
                           log_metric: bool = True) -> pd.DataFrame:
        """
        Removes rows with null values from dataframe's specified columns if the percentage of null values is less that the specified value
        
        args:
            (pandas.DataFrame) df: The DataFrame to be modified
            (int) percentage_null_values: The percentage of null values to be taken as limit
        
        Returns:
            (pandas.DataFrame) result: The resulting DataFrame
            TODO: Remove this line :(int) number_of_deleted_lines: The number of lines removed from the DataFrame
        """

        total_number_of_rows_removed = 0
        initial_number_of_rows = len(df)

        if columns in None:
            columns = df.columns

        for value in columns:
            if value in df.columns:
                percentage = df[value].isnull().sum() * 100 / len(df)
                if percentage < percentage_null_values:
                    number_of_row_before = len(df)
                    df = df[df[value].notna()]
                    number_of_rows_removed = number_of_row_before - len(df)
                    total_number_of_rows_removed = total_number_of_rows_removed + number_of_rows_removed
                    print(f"For column {value}, % missing values = {percentage}")
                    print(f"{number_of_rows_removed} rows were removed")
            else:
                print(f"Column {value} not found in DataFrame")

        print(
            f"----------------- Total number of rows removed = {total_number_of_rows_removed} on {initial_number_of_rows} -----------------")
        print(
            f"----------------- Thus {(total_number_of_rows_removed / initial_number_of_rows) * 100}% -----------------")
        print("\n")

        if log_metric:
            mlflow.log_metric("MissingValues.TrimPercentage", total_number_of_rows_removed)
        return df

    def trim_all(df: pd.DataFrame, columns: list[str], log_metric: bool = True):
        """
        Removes null values from dataframe's specific columns
        
        args:
            (pandas.DataFrame) df: The DataFrame to be modified
            (list[str]) columns: The names of the columns from which the null values should be removed
        
        Returns:
            (pandas.DataFrame) result: The resulting DataFrame
            TODO: Remove this line :(int) number_of_deleted_lines: The number of lines removed from the DataFrame
        """

        total_number_of_rows_removed = 0
        initial_number_of_rows = len(df)

        for value in columns:
            if value in df.columns:
                number_of_row_before = len(df)
                df = df[df[value].notna()]
                number_of_rows_removed = number_of_row_before - len(df)
                total_number_of_rows_removed = total_number_of_rows_removed + number_of_rows_removed
                print(f"From column {value}, {number_of_rows_removed} rows were removed")

            else:
                print(f"Column {value} not found in DataFrame")

        print(
            f"-- Total number of rows removed = {total_number_of_rows_removed} on {initial_number_of_rows}")
        print(
            f"-- Thus {(total_number_of_rows_removed / initial_number_of_rows) * 100}%")
        print("\n")
        if log_metric:
            mlflow.log_metric("MissingValues.TrimAll.DeleteCount", total_number_of_rows_removed)

        return df

    def set_default(df: pd.DataFrame, columns: list[str], default_values: list,
                    log_metric: bool = True) -> pd.DataFrame:
        # TODO: Change argument from 2 list to dict
        """
        Sets all null values from dataframe's specific columns to the default value provided as argument

        args:
            (pandas.DataFrame) df: The DataFrame to be modified
            (list[str]) columns: The names of the columns from which the null values should be removed
            (list[any]) default_values: The default values to be used for each column

        Returns:
            (pandas.DataFrame) result: The resulting DataFrame
        """

        if columns is None:
            columns = df.columns

        if len(columns) != len(default_values):
            raise Exception(f"Number of columns and default vaues don't match")

        for column in columns:
            if column not in df.columns:
                raise Exception(f"Column {column} not in the DataFrame")

            if df[column].dtype is type(default_values[columns.index(column)]):
                raise Exception(f"Types of column {column} and {default_values[columns.index(column)]} do not match")

        total_number_of_rows_updated = 0
        initial_number_of_rows = len(df)

        for column in columns:
            number_of_row_to_update = df[column].isnull().sum()
            df[column].fillna(default_values[columns.index(column)], inplace=True)
            total_number_of_rows_updated = total_number_of_rows_updated + number_of_row_to_update
            print(f"From column {column}, {number_of_row_to_update} data cells were updated")

        print(
            f"-- Total number of cells updated = {total_number_of_rows_updated} on {df.shape[0] * df.shape[1]} cells")
        print(
            f"-- Thus {(total_number_of_rows_updated / (df.shape[0] * df.shape[1])) * 100}% of the df was changed to default values")
        print("\n")
        if log_metric:
            mlflow.log_metric("MissingValues.SetDefault.UpdateCount", total_number_of_rows_updated)
        return df
