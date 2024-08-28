import mlflow
import mlflow.data
import optuna
from mlflow.data.pandas_dataset import PandasDataset

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, auc, roc_curve

from sklearn.model_selection import train_test_split

from DataPreparation.VariablePreparation import TypeConversion, Encoding
from utils import Utils, MlFlowUtils
from globals import ROOT_DIRECTORY, EXPERIMENTER, TMP_DIRECTORY, DATA_VERSION


class LogisticClassification:
    @staticmethod
    def get_data(data_path: str, target_column: str):
        # Get experiment's Data
        df = pd.read_csv(data_path)
        numeric_variables = [
            'SCORE_NOTE',
            'YEAR_NAISSANCE',
            'PARTIC_N_ENFANT',
            'YEAR_EMPLOI',
            'YEAR_HABITAT',
            'NB_ECH'
        ]

        financial_variables = [
            'C_DSA',
            'PARTIC_MT_REVENU_EURO',
            'PARTIC_MT_AUTRE_REVENU_EURO',
            'PARTIC_MT_CHARGES_EURO',
            'PARTIC_MT_AUTRES_CREDITS_EURO',
            'VAL_APPORT_TTC_EURO',
            'VAL_PL_TTC_EURO',
            'CT_AM_FIN_AMOUNT_EURO',
            'MT_MENS',
            'OA',
            'ASSUR',
            'VAL_CATALOGUE_TTC_EURO',
            'VAL_OPTION_TTC_EURO',
            'VAL_ACC_REM_TTC_EURO',
            'VAL_VR_TTC_EURO'
        ]

        consent_variables = [
            'CON_BL_POST_CHOICE',
            'CON_BL_EMAIL_CHOICE',
            'CON_BL_PHONE_CHOICE',
            'CON_BL_SMS_CHOICE',
            'CON_BL_MM_SCORE_CHOICE',
            'CON_BL_BMW_DATA_TRF_CHOICE',
            'CON_BL_DEALER_DATA_TRF_CHOICE',
            'CON_BL_CRM_SCORE_CHOICE'
        ]

        categorical_variables = [
            # Most relevant variables
            'CODE_POSTAL',
            'SITFAM_ID',
            'PROFES_ID',
            'HABITA_ID',
            'ID_MARQ_BIEN',
            'VEH_DES_ENGTYPE',
            'ID_ETAT_BIEN',
            'ID_NAT_BIEN',
            'SERIE_LIBELLE',
            'CT_ID_LP',
            'CO_Client',
            'ID_MARQ_BIEN',
            'VEH_DES_ENGTYPE',
            'ID_ETAT_BIEN',
            'ID_NAT_BIEN',
            'SERIE_LIBELLE',
            'Ancien_renouvelant',

            # Less relavant variables
            # 'ID_BAREME_FI',
        ]

        df = TypeConversion.convert_to_numeric(df=df, numeric_variables=numeric_variables + financial_variables)
        df = TypeConversion.convert_to_categorical(df=df, category_variables=categorical_variables)

        #df.to_csv(f"{ROOT_DIRECTORY}Data/V2/Processed/test_df.csv", index=False)
        X = df[[
            'SCORE_NOTE',
            'YEAR_NAISSANCE',
            'PARTIC_N_ENFANT',
            'YEAR_EMPLOI',
            'YEAR_HABITAT',
            'NB_ECH',

            'C_DSA',
            'PARTIC_MT_REVENU_EURO',
            'PARTIC_MT_AUTRE_REVENU_EURO',
            'PARTIC_MT_CHARGES_EURO',
            'PARTIC_MT_AUTRES_CREDITS_EURO',
            'VAL_APPORT_TTC_EURO',
            'VAL_PL_TTC_EURO',
            'CT_AM_FIN_AMOUNT_EURO',
            'MT_MENS',
            'ASSUR',
            'VAL_CATALOGUE_TTC_EURO',
            'VAL_OPTION_TTC_EURO',
            'VAL_ACC_REM_TTC_EURO',
            'VAL_VR_TTC_EURO',

            'CODE_POSTAL',
            'SITFAM_ID',
            'PROFES_ID',
            'HABITA_ID',
            'ID_MARQ_BIEN',
            'VEH_DES_ENGTYPE',
            'ID_ETAT_BIEN',
            'ID_NAT_BIEN',
            'SERIE_LIBELLE',
            'CT_ID_LP',
            'CO_Client',
            'ID_MARQ_BIEN',
            'VEH_DES_ENGTYPE',
            'ID_ETAT_BIEN',
            'ID_NAT_BIEN',
            'SERIE_LIBELLE',
            'Ancien_renouvelant'
        ]]

        y = df[target_column]

        for column in categorical_variables:
            if column in X.columns:
                X = Encoding.encode_onehot(X, column)

        dataset = mlflow.data.from_pandas(
            df, source=data_path, name="data", targets=target_column
        )

        return X, y, dataset

    @staticmethod



    @staticmethod
    def run_experiment():
        print("Starting execution...")
        experiment_tags = {
            "experimenter": EXPERIMENTER,
            "experiment version": 1.0,
            "data version": 2.1}

        experiment = MlFlowUtils.create_or_get_experiment(experiment_name="Logistic Regression")

        # Get data
        print("Collecting data...")
        X, y, dataset = LogisticClassification.get_data(data_path=f"{ROOT_DIRECTORY}Data/data.csv",
                                                        target_column="Renouvelant")

        # Train model
        with mlflow.start_run(experiment_id=experiment.experiment_id, tags=experiment_tags):
            # Set experiment tags
            print("Starting experiment...")
            mlflow.set_experiment_tags(experiment_tags)

            # Log experiment's dataset
            data_tags = {"version": DATA_VERSION}
            # mlflow.log_input(dataset, context="training", tags=data_tags)

            print("Splitting data...")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            print("Training model...")
            # Set and Log experiment's params
            params = {
                "penalty": "l2",
                "random_state": 0
            }
            mlflow.log_params(params)

            y_pred = None

            def objective(trial):
                model = LogisticRegression(
                    penalty=params["penalty"],
                    max_iter=trial.suggest_int("max_iter", 50, 500),
                    solver=trial.suggest_categorical("solver",["newton-cholesky"]),
                    random_state=params["random_state"]
                )

                # Train the model


                model.fit(X_train, y_train)

                y_pred_o = model.predict(X_test)

                return f1_score(y_test, y_pred_o)


            study = optuna.create_study(study_name="Logistic Regression", direction="maximize")
            study.optimize(objective, n_trials=1)

            trial = study.best_trial
            print("---------------------Trial-----------------------")
            print(trial)
            print("Logging metrices")
            mlflow.log_params(trial.params)
            mlflow.log_metrics({'f1-score': trial.value})


            print("Making predictions...")
            model_logger = LogisticRegression(
                penalty="l2",
                max_iter=trial.params["max_iter"],
                solver=trial.params["solver"],
                random_state=0

            )

            model_logger.fit(X_train, y_train)
            y_pred = model_logger.predict(X_test)

            print("Logging metrices...")
            mlflow.log_metric(key="Accuracy", value=accuracy_score(y_test, y_pred))
            
            

            """
            classificationReport = classification_report(y_true=y_test, y_pred=y_pred, output_dict=True)
            report = {
                'precision':classificationReport['precision'],
                'recall':classificationReport['recall'],
                'f1-score':classificationReport['f1-score'],
                'support':classificationReport['support']
            }
            mlflow.log_metrics(report)
            """

            precision = precision_score(y_test, y_pred, average="weighted")
            recall = recall_score(y_test, y_pred, average="weighted")
            f1 = f1_score(y_test, y_pred, average="weighted")

            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            print("Logging figures...")
            plt.figure()
            cm = confusion_matrix(y_test, y_pred)
            cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                     index=['Predict Positive:1', 'Predict Negative:0'])
            diagram = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
            diagram.figure.savefig(TMP_DIRECTORY + "Logistic regression Confusion Matrix.png")
            mlflow.log_artifact(TMP_DIRECTORY + "Logistic regression Confusion Matrix.png",
                                artifact_path="Confusion matrix")

            plt.figure()
            coefficients = model_logger.coef_[0]
            feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefficients)})
            feature_importance = feature_importance.sort_values('Importance', ascending=True).nlargest(20, 'Importance')
            diagram2 = feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
            diagram2.figure.savefig(TMP_DIRECTORY + "Logistic regression feature importance.png")
            mlflow.log_artifact(TMP_DIRECTORY + "Logistic regression feature importance.png",
                                artifact_path="Feature importance")

            """
            """

        print("Execution successful")



    @staticmethod
    def create_variable():
        experiment_tags = {
            "experimenter": "Ulrich KAM",
            "experiment version": 1.0,
            "data version": 2.1}
        experiment = MlFlowUtils.create_or_get_experiment(experiment_name="Logistic Regression")

        # Get experiment's Data
        money_columns = [
            'C_DSA',
            'PARTIC_MT_REVENU_EURO',
            'PARTIC_MT_AUTRE_REVENU_EURO',
            'PARTIC_MT_CHARGES_EURO',
            'PARTIC_MT_AUTRES_CREDITS_EURO',
            'VAL_APPORT_TTC_EURO',
            'VAL_PL_TTC_EURO',
            'CT_AM_FIN_AMOUNT_EURO',
            'MT_MENS',
            'ASSUR',
            'VAL_CATALOGUE_TTC_EURO',
            'VAL_OPTION_TTC_EURO',
            'VAL_ACC_REM_TTC_EURO',
            'VAL_VR_TTC_EURO'
        ]

        df = pd.read_csv(f"{ROOT_DIRECTORY}Data/V3/Processed/data_train.csv", low_memory=False)
        X = df[money_columns]
        y = df["Renouvelant"]

        # Train model
        with mlflow.start_run(experiment_id=experiment.experiment_id, tags=experiment_tags):
            # Set experiment tags
            mlflow.set_experiment_tags(experiment_tags)

            # Log experiment's dataset
            data_tags = {"version": "1.0.0"}

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

            # Set and Log experiment's params
            params = {
                "penalty": "l2",
                "max_iter": 200,
                "solver": 'liblinear',
            }
            mlflow.log_params(params)

            model = LogisticRegression(
                penalty=params["penalty"],
                max_iter=params["max_iter"],
                solver=params["solver"],
            )

            # Train the model
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)

            mlflow.log_metric(key="Accuracy", value=accuracy_score(y_test, y_pred_test))
            mlflow.log_metric(key="F1-score", value=f1_score(y_test, y_pred_test))

            print("Logging figures...")
            plt.figure()
            cm = confusion_matrix(y_test, y_pred_test)
            cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                     index=['Predict Positive:1', 'Predict Negative:0'])
            diagram = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
            diagram.figure.savefig(TMP_DIRECTORY + "/Logistic regression Confusion Matrix.png")
            mlflow.log_artifact(TMP_DIRECTORY + "/Logistic regression Confusion Matrix.png",
                                artifact_path="Confusion matrix")

            plt.figure()
            coefficients = model.coef_[0]
            feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': np.abs(coefficients)})
            feature_importance = feature_importance.sort_values('Importance', ascending=True).nlargest(20, 'Importance')
            diagram2 = feature_importance.plot(x='Feature', y='Importance', kind='barh', figsize=(10, 6))
            diagram2.figure.savefig(TMP_DIRECTORY + "/Logistic regression feature importance.png")
            mlflow.log_artifact(TMP_DIRECTORY + "/Logistic regression feature importance.png",
                                artifact_path="Feature importance")
            # if log_artefacts:
            # mlflow.log_artifact(TMP_DIRECTORY + f"/Logistic regression Confusion Matrix.png",
            # artifact_path=f"Logistic classification confusion matrix")

            # mlflow.log_metrics(classification_report(y_test, y_pred_test))
        print("First execution successful")