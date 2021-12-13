from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import torch

import mlflow

from torch import nn
from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/dhkim92@icloud.com/MLflow_test1")


data = load_iris()
X, y = data.data, data.target
seed = 10  # Specify a seed value.

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=seed, stratify=y)


current_run = mlflow.start_run()

# Logging the seed value to passed to the train_test_split function.
mlflow.log_param("seed", seed)

estimators = int(input("Estimator(s): "))

# Model definition.
rclf = RandomForestClassifier(n_estimators=estimators)

mlflow.sklearn.autolog()
rclf.fit(X_train, y_train)
metrics = mlflow.sklearn.eval_and_log_metrics(
    rclf, X_test, y_test, prefix="val_")
mlflow.end_run()

PATH = "/content/drive/MyDrive/gh/datascience/dacon/job_care_clf/dataset"
train_df = pd.read_csv(PATH + "/train.csv")
test_df = pd.read_csv(PATH + "/test.csv")
sample_submission = pd.read_csv(PATH + "/sample_submission.csv")

train = train_df[train_df['contents_open_dt'].apply(
    lambda x: pd.Timestamp(x).month) < 11].copy()
val = train_df[train_df['contents_open_dt'].apply(
    lambda x: pd.Timestamp(x).month) == 11].copy()
test = test_df.copy()

for df in [train, val, test]:
    df.drop(['contents_open_dt', 'contents_rn', 'id', 'person_rn', 'contents_open_dt',
            'person_prefer_f', 'person_prefer_g'], axis=1, inplace=True)

columns = sorted(test.columns)
train = train[columns+['target']]*1
val = val[columns+['target']]*1
test = test[columns]*1

cat_idxs = []
cat_dims = []
for idx, col in enumerate(train.columns):
    if 'match' not in col and col != 'target':
        le = LabelEncoder()
        le.fit(train_df[col].values)
        le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
        train[col] = train[col].apply(lambda x: le_dict.get(x, len(le_dict)))
        val[col] = val[col].apply(lambda x: le_dict.get(x, len(le_dict)))
        test[col] = test[col].apply(lambda x: le_dict.get(x, len(le_dict)))
        cat_idxs.append(idx)
        cat_dims.append(len(le_dict)+1)

X_train = train.drop('target', axis=1).values
y_train = train['target'].values
X_val = val.drop('target', axis=1).values
y_val = val['target'].values
X_test = test.values
eval_set = (X_val, y_val)

clf = TabNetClassifier(cat_idxs=cat_idxs,
                       cat_dims=cat_dims,
                       cat_emb_dim=3,
                       optimizer_fn=torch.optim.RAdam,  # Any optimizer works here
                       mask_type='entmax',  # "sparsemax",
                       )


class F1_Score(Metric):
    def __init__(self):
        self._name = "f1"
        self._maximize = True

    def __call__(self, y_true, y_score):
        score = f1_score(y_true, (y_score[:, 1] > 0.5)*1)
        return score


#
# callback for MLflow integration
#
class MLCallback(Callback):
    def on_train_begin(self, logs=None):

        mlflow.set_tracking_uri('databricks')
        mlflow.set_experiment('/Users/dhkim92@icloud.com/MLflow_test1')

        mlflow.start_run(run_name='tabnet_test')

        mlflow.log_params({"seed": 1})

    def on_train_end(self, logs=None):

        mlflow.end_run()

    def on_epoch_end(self, epoch, logs=None):

        # send to MLFlow
        mlflow.log_metric("train_f1", logs['train_f1'])
        mlflow.log_metric("val_f1", logs["val_f1"])
        mlflow.log_metric("train_logloss", logs['train_logloss'])
        mlflow.log_metric("val_logloss", logs["val_logloss"])


mlflow.end_run()
mlcbck = MLCallback()

clf.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    eval_name=['train', 'val'],
    eval_metric=['logloss', 'f1'],
    max_epochs=100, patience=10,
    batch_size=1024,
    virtual_batch_size=256,
    num_workers=1,
    drop_last=False,
    callbacks=[mlcbck]
)

preds = clf.predict_proba(X_test)
preds = (preds[:, 1] > 0.5)*1
