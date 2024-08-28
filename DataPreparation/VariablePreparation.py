import pandas as pd
from enum import Enum
from pandas import CategoricalDtype
from sklearn.preprocessing import MinMaxScaler

class Encoding(Enum):
    ONEHOT = 1
    ORDINAL = 2
    TARGET = 3
    BINARY = 4

    # LABEL = 5
    # DUMMY = 6
    # COUNT = 7

    @staticmethod
    def encode_onehot(df: pd.DataFrame, column: str) -> pd.DataFrame:
        encoded_df = pd.get_dummies(df, columns=[column], dtype=int)
        return encoded_df

    @staticmethod
    def encode_ordinal(df: pd, column: str, category_map: map) -> pd.DataFrame:
        if category_map is None:
            category_map = {}
            i = 1
            for value in df[column].unique():
                category_map[f"{value}"] = i
                i = i + 1

        df[column] = df[column].map(category_map)
        return df

    @staticmethod
    def encode_target(df: pd.DataFrame, column: str, target_column: str):
        target_mean = df.groupby(column)[target_column].mean()
        df[column] = df[column].map(target_mean)
        return df

    @staticmethod
    def encode_binary(df: pd.DataFrame, column: str, category_map: map):
        result = Encoding.encode_ordinal(df, column, category_map)
        result[column] = result[column].apply(lambda x: format(x, 'b'))
        return result


class TypeConversion:
    @staticmethod
    def convert_to_numeric(df: pd.DataFrame, numeric_variables: list[str]) -> pd.DataFrame:
        for numeric_variable in numeric_variables:
            if df[numeric_variable].dtype != float:
                df[numeric_variable] = df[numeric_variable].astype(float)

        return df

    @staticmethod
    def convert_to_categorical(df: pd.DataFrame, category_variables: list[str], ordered: dict = None) -> pd.DataFrame:
        for category_variable in category_variables:
            if df[category_variable].dtype != 'object':
                df[category_variable] = df[category_variable].astype(str)

            if ordered is not None:
                if category_variable in ordered.keys():
                    cat_type = CategoricalDtype(categories=ordered[category_variable], ordered=True)
                    df[category_variable] = df[category_variable].astype(cat_type)

            else:
                df[category_variable] = df[category_variable].astype("category")

        return df

    @staticmethod
    def convert_to_string(df: pd.DataFrame, string_variables: list[str]) -> pd.DataFrame:
        for string_variable in string_variables:
            if df[string_variable].dtype != 'object':
                df[string_variable] = df[string_variable].astype(str)

        return df

    @staticmethod
    def sum_columns(df: pd.DataFrame, column_name: str, column_names: list[str]) -> pd.DataFrame:
        df[column_name] = df[column_names].sum(axis=1)
        df.drop(labels=column_names, axis="columns", inplace=True)
        return df

class Normalizer():
    @staticmethod
    def min_max_normalize(df: pd, column_names: list[str]) -> pd.DataFrame:
        scaler = MinMaxScaler()
        for column in column_names:
            df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
        return df