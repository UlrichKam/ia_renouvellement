import mlflow
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from pandas import CategoricalDtype
import time
from sklearn.inspection import permutation_importance
import DataPreparation
from DataPreparation.VariablePreparation import TypeConversion, Normalizer
from utils import Utils, MlFlowUtils
from globals import ROOT_DIRECTORY, EXPERIMENTER, TMP_DIRECTORY
from catboost import CatBoostClassifier
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, auc, roc_curve
import optuna


class CatBoostModel:

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
            'Consent',
            'CODE_POSTAL',
            'SITFAM_ID',
            'PROFES_ID',
            'HABITA_ID',
            'VEH_DES_ENGTYPE',
            'SERIE_LIBELLE',
            'CT_ID_LP',
            'CO_Client',
            'ID_MARQ_BIEN',
            'ID_ETAT_BIEN',
            'ID_NAT_BIEN',

            # Less relavant variables
            # 'ID_BAREME_FI',
        ]

        #df = TypeConversion.convert_to_numeric(df=df, numeric_variables=numeric_variables + financial_variables)
        df = TypeConversion.convert_to_categorical(df=df, category_variables=categorical_variables)

        # df.to_csv(f"{ROOT_DIRECTORY}Data/V2/Processed/test_df.csv", index=False)
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

            'Consent',
            'CODE_POSTAL',
            'SITFAM_ID',
            'PROFES_ID',
            'HABITA_ID',
            'VEH_DES_ENGTYPE',
            'ID_ETAT_BIEN',
            'ID_NAT_BIEN',
            'SERIE_LIBELLE',
            'CT_ID_LP',
            'CO_Client',
            'ID_MARQ_BIEN',
            'Ancien_renouvelant'
        ]]

        y = df[target_column]

        return X, y

    @staticmethod
    def run_experiment():
        print("Starting execution...")
        # Create an experiment if it doesn't exist
        experiment_tags = {
            "experimenter": EXPERIMENTER,
            "experiment version": 1.0,
            "data version": 2.0}
        experiment = MlFlowUtils.create_or_get_experiment(experiment_name="CatBoost")

        # Get data
        print("Collecting data...")
        X, y = CatBoostModel.get_data(data_path=f"{ROOT_DIRECTORY}Data/data.csv",
                                               target_column="Renouvelant")

        # Train model
        with mlflow.start_run(experiment_id=experiment.experiment_id, tags=experiment_tags):
            print("Starting experiment...")
            # Set experiment tags
            #mlflow.set_experiment_tags(experiment_tags)

            df = pd.read_csv(f"{ROOT_DIRECTORY}Data/data.csv", low_memory=False)
            df = Normalizer.min_max_normalize(df, [
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
            ])


            df = TypeConversion.convert_to_categorical(df=df, category_variables=[
                'Consent',
                'CODE_POSTAL',
                'SITFAM_ID',
                'PROFES_ID',
                'HABITA_ID',
                'VEH_DES_ENGTYPE',
                'SERIE_LIBELLE',
                'CT_ID_LP',
                'CO_Client',
                'ID_MARQ_BIEN',
                'ID_ETAT_BIEN',
                'ID_NAT_BIEN',
                'Ancien_renouvelant'])

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

                'Consent',
                'CODE_POSTAL',
                'SITFAM_ID',
                'PROFES_ID',
                'HABITA_ID',
                'VEH_DES_ENGTYPE',
                'ID_ETAT_BIEN',
                'ID_NAT_BIEN',
                #'SERIE_LIBELLE',
                'CT_ID_LP',
                'CO_Client',
                'ID_MARQ_BIEN',
                'Ancien_renouvelant'
            ]]

            y = df["Renouvelant"]

            cat_features = [
                # Most relevant variables
                'Consent',
                'CODE_POSTAL',
                'SITFAM_ID',
                'PROFES_ID',
                'HABITA_ID',
                'VEH_DES_ENGTYPE',
                #'SERIE_LIBELLE',
                'CT_ID_LP',
                'CO_Client',
                'ID_MARQ_BIEN',
                'ID_ETAT_BIEN',
                'ID_NAT_BIEN',
                'Ancien_renouvelant',

                # Less relavant variables
                # 'ID_BAREME_FI',
            ]

            print("Splitting data...")
            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=123)

            def objective(trial):
                model = CatBoostClassifier(

                    iterations=trial.suggest_int("iterations", 100, 1000),
                    learning_rate=trial.suggest_float("learning_rate", 1e-3, 1e-1, log=True),
                    depth=trial.suggest_int("depth", 4, 10),
                    l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-8, 100.0, log=True),
                    bootstrap_type=trial.suggest_categorical("bootstrap_type", ["Bayesian"]),
                    random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
                    bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 10.0),
                    od_type=trial.suggest_categorical("od_type", ["IncToDec", "Iter"]),
                    od_wait=trial.suggest_int("od_wait", 10, 50),
                    verbose=False
                )
                model.fit(train_X, train_y, cat_features)
                pred_y = model.predict(test_X)
                return f1_score(test_y, pred_y)

            study = optuna.create_study(study_name="catboost", direction="maximize")
            start_time = time.time()
            study.optimize(objective, n_trials=25, n_jobs=-1, gc_after_trial=True)
            end_time = time.time()
            training_time = end_time - start_time
            # Log training time as a metric
            mlflow.log_metric("training_time_seconds", training_time)
            trial = study.best_trial
            mlflow.log_params(trial.params)

            model_best = CatBoostClassifier(
                iterations=trial.params["iterations"],
                learning_rate=trial.params["learning_rate"],
                depth=trial.params["depth"],
                l2_leaf_reg=trial.params["l2_leaf_reg"],
                bootstrap_type=trial.params["bootstrap_type"],
                random_strength=trial.params["random_strength"],
                bagging_temperature=trial.params["bagging_temperature"],
                od_type=trial.params["od_type"],
                od_wait=trial.params["od_wait"],
                verbose=False
            )

            model_best.fit(train_X, train_y, cat_features)
            y_pred = model_best.predict(test_X)

            print("Logging metrices...")
            precision = precision_score(test_y, y_pred, average="weighted")
            recall = recall_score(test_y, y_pred, average="weighted")
            f1 = f1_score(test_y, y_pred, average="weighted")

            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)

            print("Logging figures...")
            plt.figure()
            cm = confusion_matrix(test_y, y_pred)
            cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                     index=['Predict Positive:1', 'Predict Negative:0'])
            diagram = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
            diagram.figure.savefig(TMP_DIRECTORY + "CatBoost Confusion Matrix.png")
            #mlflow.log_artifact(TMP_DIRECTORY + "CatBoost Confusion Matrix.png", artifact_path="")

            plt.figure()
            perm_importance = permutation_importance(model_best, test_X, test_y, n_repeats=10, random_state=1066)
            sorted_idx = perm_importance.importances_mean.argsort()
            fig = plt.figure(figsize=(12, 6))
            plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
            plt.yticks(range(len(sorted_idx)), np.array(test_X.columns)[sorted_idx])
            plt.title('Permutation Importance')
            plt.savefig(TMP_DIRECTORY + "CatBoost permutation importance.png")
            #mlflow.log_artifact(TMP_DIRECTORY + "CatBoost permutation importance.png", artifact_path="")

            plt.figure()
            feature_importance = model_best.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            fig = plt.figure(figsize=(12, 6))
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
            plt.yticks(range(len(sorted_idx)), np.array(test_X.columns)[sorted_idx])
            plt.title('Feature Importance')
            plt.savefig(TMP_DIRECTORY + "CatBoost feature importance.png")
            #mlflow.log_artifact(TMP_DIRECTORY + "CatBoost feature importance.png", artifact_path="")

            if len(set(test_y)) == 2:
                y_prob = model_best.predict_proba(test_X)[:, 1]
                fpr, tpr, thresholds = roc_curve(test_y, y_prob)
                roc_auc = auc(fpr, tpr)

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
                plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic')
                plt.legend(loc="lower right")

                plt.savefig(f"{TMP_DIRECTORY}CatBoost - roc_curve.png", format='png')
                #mlflow.log_artifact(f"{TMP_DIRECTORY}CatBoost - roc_curve.png", "")
                plt.close()
