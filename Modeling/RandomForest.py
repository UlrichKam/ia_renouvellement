import io

import numpy as np
from sklearn.inspection import permutation_importance

from globals import EXPERIMENTER, ROOT_DIRECTORY, TMP_DIRECTORY
import time
import mlflow
import pandas as pd
from utils import Utils, MlFlowUtils
from globals import ROOT_DIRECTORY, EXPERIMENTER
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, auc, roc_curve
from sklearn.model_selection import train_test_split
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


class RandomForest:
    @staticmethod
    def run_experiment():
        experiment_tags = {
            "experimenter": EXPERIMENTER,
            "experiment version": 1.0,
            "data version": 3.0}
        experiment = MlFlowUtils.create_or_get_experiment(experiment_name="RandomForest")

        df = pd.read_csv(ROOT_DIRECTORY + "Data/data.csv", encoding='latin1')

        with mlflow.start_run(experiment_id=experiment.experiment_id, tags=experiment_tags):

            print('Experiment started')
            mlflow.set_experiment_tags(experiment_tags)
            print('Pré-traitement des données')
            df['ID_BAREME_FI'] = df['ID_BAREME_FI'].astype('category')
            df['CT_ID_LP'] = df['CT_ID_LP'].astype('category')
            df['SCORE_NOTE'] = df['SCORE_NOTE'].astype(int)
            df['CODE_POSTAL'] = df['CODE_POSTAL'].astype('category')
            df['SITFAM_ID'] = df['SITFAM_ID'].astype('category')
            df['YEAR_NAISSANCE'] = df['YEAR_NAISSANCE'].astype(int)
            df['NB_ECH'] = df['NB_ECH'].astype(int)
            df['YEAR_EMPLOI'] = df['YEAR_EMPLOI'].astype(int)
            df['MT_MENS'] = df['MT_MENS'].astype(int)
            df['YEAR_HABITAT'] = df['YEAR_HABITAT'].astype(int)
            df['PROFES_ID'] = df['PROFES_ID'].astype('category')
            df['HABITA_ID'] = df['HABITA_ID'].astype('category')
            df['ID_MARQ_BIEN'] = df['ID_MARQ_BIEN'].astype('category')
            df['ID_ETAT_BIEN'] = df['ID_ETAT_BIEN'].astype('category')
            df['ID_NAT_BIEN'] = df['ID_NAT_BIEN'].astype('category')
            df['SERIE_LIBELLE'] = df['SERIE_LIBELLE'].astype('category')
            df['CO_Client'] = df['CO_Client'].astype('category')
            df['Ancien_renouvelant'] = df['Ancien_renouvelant'].astype(int)
            df['Consent'] = df['Consent'].astype(int)
            df['VEH_DES_ENGTYPE'] = df['VEH_DES_ENGTYPE'].astype('category')

            print("Encodage des variables")
            le = LabelEncoder()
            df['ID_BAREME_FI'] = le.fit_transform(df['ID_BAREME_FI'])
            df['CT_ID_LP'] = le.fit_transform(df['CT_ID_LP'])
            df['SITFAM_ID'] = le.fit_transform(df['SITFAM_ID'])
            df['PROFES_ID'] = le.fit_transform(df['PROFES_ID'])
            df['HABITA_ID'] = le.fit_transform(df['HABITA_ID'])
            df['ID_MARQ_BIEN'] = le.fit_transform(df['ID_MARQ_BIEN'])
            df['ID_ETAT_BIEN'] = le.fit_transform(df['ID_ETAT_BIEN'])
            df['SERIE_LIBELLE'] = le.fit_transform(df['SERIE_LIBELLE'])
            df['CO_Client'] = le.fit_transform(df['CO_Client'])
            df['VEH_DES_ENGTYPE'] = le.fit_transform(df['VEH_DES_ENGTYPE'])

            print('Anonymisation des données')

            del df['ID_CONT']

            # Load the data
            X, y = df[['SCORE_NOTE','YEAR_NAISSANCE','NB_ECH','YEAR_EMPLOI','MT_MENS','YEAR_EMPLOI','MT_MENS','YEAR_HABITAT','Ancien_renouvelant','Consent']], df['Renouvelant']
            print('Préparation des données')

            for column in df.columns:
                if df[column].dtype is pd.CategoricalDtype():
                    df[column] = LabelEncoder().fit_transform(df[column])
            # Splitting

            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=123)

            print('Entrainement du modèle et optimisation des Hyperparamètres')
            """
            model = RandomForestClassifier(
                # random_state=trialValue.suggest_int("random_state", 20, 50),
                # n_estimators=trialValue.suggest_int("n_estimators", 100, 1000),
                # max_features=trialValue.suggest_int("max_features", 1, 20),
                # min_samples_leaf=trialValue.suggest_int("min_samples_leaf", 5, 200),
                # n_jobs=trialValue.suggest_int("n_jobs", 1, 20),
                # max_depth=trialValue.suggest_int("max_depth", 5, 30)
            )

            model.fit(train_X, train_y)

            """
            def objective(trialValue):
                model = RandomForestClassifier(
                    random_state=trialValue.suggest_int("random_state", 20, 50),
                    n_estimators=trialValue.suggest_int("n_estimators", 100, 1000),
                    max_features=trialValue.suggest_int("max_features", 1, 20),
                    min_samples_leaf=trialValue.suggest_int("min_samples_leaf", 5, 200),
                    n_jobs=trialValue.suggest_int("n_jobs", 1, 20),
                    max_depth=trialValue.suggest_int("max_depth", 5, 30)
                )
                
                model.fit(train_X, train_y)
                prediction_y = model.predict(test_X)
                return f1_score(test_y, prediction_y, average="weighted")




            study = optuna.create_study(direction="maximize")

            start_time = time.time()
            study.optimize(objective, n_trials=4)
            end_time = time.time()
            print('Modèle entrainé')
            training_time = end_time - start_time
            # Log training time as a metric
            mlflow.log_metric("training_time_seconds", training_time)

            trial = study.best_trial

            print('Enrégistrement dans MLflow')
            mlflow.log_params(trial.params)

            best_model = RandomForestClassifier(
                random_state=trial.params['random_state'],
                n_estimators=trial.params['n_estimators'],
                max_features=trial.params['max_features'],
                min_samples_leaf=trial.params['min_samples_leaf'],
                n_jobs=trial.params['n_jobs'],
                max_depth=trial.params['max_depth']
            )


            best_model.fit(train_X, train_y)
            pred_y = best_model.predict(test_X)
            print("Logging best model's performances metrics")

            precision = precision_score(test_y, pred_y, average="weighted")
            recall = recall_score(test_y, pred_y, average="weighted")
            f1 = f1_score(test_y, pred_y, average="weighted")

            #model_classification_report = classification_report(test_y, pred_y, target_names=['Class 0', 'Class 1'])
            """for key, value in model_classification_report.items():
                if isinstance(value, dict):
                    for metric_name, metric_value in value.items():
                        mlflow.log_metric(f"{key}_{metric_name}", metric_value)
                else:
                    mlflow.log_metric(key, value)
            """
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1", f1)


            plt.figure()
            cm = confusion_matrix(test_y, pred_y)
            cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],
                                     index=['Predict Positive:1', 'Predict Negative:0'])
            diagram = sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
            diagram.figure.savefig(TMP_DIRECTORY + "Random forest Confusion Matrix.png")
            mlflow.log_artifact(TMP_DIRECTORY + "Random forest Confusion Matrix.png")

            plt.figure()
            perm_importance = permutation_importance(best_model, test_X, test_y, n_repeats=10, random_state=1066)
            sorted_idx = perm_importance.importances_mean.argsort()
            fig = plt.figure(figsize=(12, 6))
            plt.barh(range(len(sorted_idx)), perm_importance.importances_mean[sorted_idx], align='center')
            plt.yticks(range(len(sorted_idx)), np.array(test_X.columns)[sorted_idx])
            plt.title('Permutation Importance')
            plt.savefig(TMP_DIRECTORY + "Random forest permutation importance.png")
            mlflow.log_artifact(TMP_DIRECTORY + "Random forest permutation importance.png")

            plt.figure()
            feature_importance = best_model.feature_importances_
            sorted_idx = np.argsort(feature_importance)
            fig = plt.figure(figsize=(12, 6))
            plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
            plt.yticks(range(len(sorted_idx)), np.array(test_X.columns)[sorted_idx])
            plt.title('Feature Importance')
            plt.savefig(TMP_DIRECTORY + "Random forest feature importance.png")
            mlflow.log_artifact(TMP_DIRECTORY + "Random forest feature importance.png")
            
            if len(set(test_y)) == 2:
                y_prob = best_model.predict_proba(test_X)[:, 1]
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

                plt.savefig(f"{TMP_DIRECTORY}Random forest - roc_curve.png", format='png')
                mlflow.log_artifact( f"{TMP_DIRECTORY}Random forest - roc_curve.png", "")
                plt.close()
