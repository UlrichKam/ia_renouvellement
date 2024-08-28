import os
import mlflow
import pandas as pd

from DataPreparation.VariablePreparation import TypeConversion
from Modeling.RandomForest import RandomForest
from Modeling.CatBoost import CatBoostModel
from globals import ROOT_DIRECTORY, EXPERIMENTER, DATA_VERSION
from utils import MlFlowUtils, Utils

from DataPreparation.MissingValues import MissingValues
from DataPreparation.Outliers import Outliers

from ExploratoryDataAnalysis.StatisticalTests import SpearmanTest, ChiSquare

from Modeling.XGBoost import XGBoostClassifier
from Modeling.LogisticClassification import LogisticClassification


dirname = os.path.dirname(__file__)


data_directory = "/Data"
def DataPreparation():
    experiment_name = "Missing values"
    experiment_tags = {
        "experimenter": EXPERIMENTER,
        "version": "2.0.0",
        "data_version": DATA_VERSION
    }

    exp = MlFlowUtils.create_or_get_experiment(experiment_name)

    with mlflow.start_run(experiment_id=exp.experiment_id):
        mlflow.set_tags(tags=experiment_tags)
        df = pd.read_csv(ROOT_DIRECTORY + f"{data_directory}/Database_Renouvelant_contrat.csv", encoding='latin1')

        MissingValues.visualise_missing(df=df, log_path_argument="/Raw")

        consent_list = [
            'CON_BL_POST_CHOICE',
            'CON_BL_EMAIL_CHOICE',
            'CON_BL_PHONE_CHOICE',
            'CON_BL_SMS_CHOICE',
            'CON_BL_MM_SCORE_CHOICE',
            'CON_BL_BMW_DATA_TRF_CHOICE',
            'CON_BL_DEALER_DATA_TRF_CHOICE',
            'CON_BL_CRM_SCORE_CHOICE'
        ]

        appended_list = [
            'CON_BL_POST_CHOICE',
            'CON_BL_EMAIL_CHOICE',
            'CON_BL_PHONE_CHOICE',
            'CON_BL_SMS_CHOICE',
            'CON_BL_MM_SCORE_CHOICE',
            'CON_BL_BMW_DATA_TRF_CHOICE',
            'CON_BL_DEALER_DATA_TRF_CHOICE',
            'CON_BL_CRM_SCORE_CHOICE',
            'YEAR_EMPLOI',
            'OA',
            'VAL_ARGUS_BIEN_EURO',
            'YEAR_HABITAT'
        ]

        df = MissingValues.trim_all(df, [elem for elem in df.columns.to_list() if elem not in appended_list])
        df = MissingValues.set_default(df, columns=consent_list, default_values=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        MissingValues.visualise_missing(df, log_path_argument="/Processed")

        money_columns = [
            'PARTIC_MT_REVENU_EURO',
            'PARTIC_MT_AUTRE_REVENU_EURO',
            'PARTIC_MT_CHARGES_EURO',
            'PARTIC_MT_AUTRES_CREDITS_EURO',
            'SCORE_NOTE',
            'C_DSA',
            'NB_ECH',
            'ASSUR',
            'VAL_OPTION_TTC_EURO',
            'VAL_VR_TTC_EURO',
            'YEAR_EMPLOI',
            'YEAR_HABITAT',
            'YEAR_NAISSANCE'
        ]

        Outliers.visualise_outliers(
            df=df,
            columns=money_columns,
            log_path_argument="/Raw"
        )

        df = Outliers.cap_outliers(df=df, columns=money_columns, upper_bound=0.99, lower_bound=0.01)

        Outliers.visualise_outliers(
            df=df,
            columns=money_columns,
            log_path_argument="/Processed"
        )

        ### Sum all consents to get a single numeric value for consents
        df = TypeConversion.sum_columns(df, "Consent", consent_list)

        ### NExt to do is normalize money values
        # Maybe each one individually, or as two values (incomes and outcomes)
        # Maybe create a simple regression with target encoding to create a correlation between numeric values (ordered by business value) to the target value


        train_data, validation_data = Utils.split_data(df)
        df.to_csv(f"{ROOT_DIRECTORY}{data_directory}/Processed/data.csv", index=False)
        train_data.to_csv(f"{ROOT_DIRECTORY}{data_directory}/Processed/data_train.csv", index=False)
        validation_data.to_csv(f"{ROOT_DIRECTORY}{data_directory}/Processed/data_validation.csv", index=False)
        print(mlflow.active_run())


def Statistical_Analysis():
    experiment_name = "Statistical Analysis"
    experiment_tags = {
        "experimenter": EXPERIMENTER,
        "version": "1.0.0",
        "data_version": DATA_VERSION
    }

    exp = MlFlowUtils.create_or_get_experiment(experiment_name)
    # mlflow.delete_experiment(exp.experiment_id)

    df = pd.read_csv(f"{ROOT_DIRECTORY}Data/data.csv")

    with mlflow.start_run(experiment_id=exp.experiment_id):
        mlflow.set_tags(tags=experiment_tags)
        SpearmanTest(df, "YEAR_NAISSANCE", "Renouvelant", 1)
        SpearmanTest(df, "PARTIC_N_ENFANT", "Renouvelant", 2)
        SpearmanTest(df, "PARTIC_MT_REVENU_EURO", "Renouvelant", 3)
        SpearmanTest(df, "PARTIC_MT_AUTRE_REVENU_EURO", "Renouvelant", 4)
        SpearmanTest(df, "PARTIC_MT_CHARGES_EURO", "Renouvelant", 5)
        SpearmanTest(df, "PARTIC_MT_AUTRES_CREDITS_EURO", "Renouvelant", 6)

        ChiSquare(df, "Ancien_renouvelant", "Renouvelant", 1)
        ChiSquare(df, "Consent", "Renouvelant", 2)
        ChiSquare(df, "CODE_POSTAL", "Renouvelant", 3)
        ChiSquare(df, "SITFAM_ID", "Renouvelant", 4)
        ChiSquare(df, "HABITA_ID", "Renouvelant", 5)
        ChiSquare(df, "VEH_DES_ENGTYPE", "Renouvelant", 6)


def XGBoostClassification():
    experiment_name = "XGBoost Classifier"
    experiment_tags = {
        "experimenter": EXPERIMENTER,
        "version": "1.0.0",
        "data_version": DATA_VERSION
    }

    exp = MlFlowUtils.create_or_get_experiment(experiment_name)
    print(exp.name)

    with mlflow.start_run(experiment_id=exp.experiment_id):
        mlflow.set_tags(tags=experiment_tags)
        df = pd.read_csv(f"{ROOT_DIRECTORY}Data/data.csv")

        xgboost = XGBoostClassifier(df=df, target_column="Renouvelant")

        xgboost.PrepareData()
        xgboost.TrainModel()

        #xgboost.



def main():
    #DataPreparation()
    #Statistical_Analysis()
    #XGBoostClassification()

    #LogisticClassification.run_experiment()
    CatBoostModel.run_experiment()
    #LogisticClassification.create_variable()
    #RandomForest.run_experiment()

if __name__ == "__main__":
    main()
