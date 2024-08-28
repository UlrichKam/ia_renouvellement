import mlflow
import pandas as pd
from globals import ROOT_DIRECTORY


class Utils:
    def read_file(path: str) -> pd.DataFrame:
        if ".xls" in path:
            return pd.read_excel(path)

        elif ".csv" in path:
            return pd.read_csv(path)

        elif ".json" in path:
            return pd.read_json(path)

    def split_data(df: pd.DataFrame):
        random_seed = 42
        sample_size = int(len(df) * 0.1)
        sample_df = df.sample(n=sample_size, random_state=random_seed)
        remaining_df = df.drop(sample_df.index)
        return remaining_df, sample_df


class MlFlowUtils:
    def create_or_get_experiment(experiment_name: str):
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            # Create experiment
            experiment_id = mlflow.create_experiment(name=experiment_name)
            experiment = mlflow.get_experiment(experiment_id)

            return experiment

        else:
            return experiment


