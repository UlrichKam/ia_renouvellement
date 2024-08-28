import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, auc, roc_curve


import mlflow
from mlflow.data.pandas_dataset import PandasDataset

class XGBoostClassifier:
    def __init__(self, df: pd.DataFrame, target_column: str):
        if target_column is None:
            raise Exception("target_column cannot be None")
        self.df = df
        self.target_column = target_column

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.y_train_encoded = None
        self.y_test_encoded = None

    def PrepareData(self):

        # Extract the features and target data separately
        X = self.df.drop(self.df[self.target_column], axis=1)
        y = self.df[self.target_column]

        # Split the data into training and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.33, random_state=17)

        # Create a label encoder object
        le = LabelEncoder()

        # Fit and transform the target variable
        self.y_train_encoded = le.fit_transform(self.y_train)
        self.y_test_encoded = le.transform(self.y_test)

    def TrainModel(self, log_metrics: bool = True):
        # Fit an XGBoost binary classifier on the training data split
        model = xgboost.XGBClassifier().fit(self.X_train, self.y_train_encoded)

        # Build the Evaluation Dataset from the test set
        y_test_pred = model.predict(X=self.X_test)

        precision = precision_score(self.y_test, y_test_pred, average="weighted")
        recall = recall_score(self.y_test, y_test_pred, average="weighted")
        f1 = f1_score(self.y_test, y_test_pred, average="weighted")

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        eval_data = self.X_test
        eval_data["label"] = self.y_test

        # Assign the decoded predictions to the Evaluation Dataset
        le = LabelEncoder()
        eval_data["predictions"] = le.inverse_transform(y_test_pred)

        # Create the PandasDataset for use in mlflow evaluate
        pd_dataset = mlflow.data.from_pandas(
            eval_data, predictions="predictions", targets="label"
        )

        if log_metrics:
            mlflow.log_input(pd_dataset, context="training")
            mlflow.xgboost.log_model(
                artifact_path="white-wine-xgb", xgb_model=model, input_example=self.X_test
            )
            result = mlflow.evaluate(data=pd_dataset, predictions=None, model_type="classifier")
