import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from Constants import *
import pickle
import time
import os

class LogisticRegressionClassifier:
  def __init__(self, x_train, x_val, x_test, label_train, label_val, label_test, attack_train, attack_val, attack_test) -> None:
    self.x_train = x_train
    self.x_val = x_val
    self.x_test = x_test
    self.label_train = label_train
    self.label_val = label_val
    self.label_test = label_test
    self.attack_train = attack_train
    self.attack_val = attack_val
    self.attack_test = attack_test

    scalar = StandardScaler()
    self.x_train_scaled = scalar.fit_transform(self.x_train)
    self.x_val_scaled = scalar.transform(self.x_val)
    self.x_test_scaled = scalar.transform(self.x_test)

    # hyperparamters for training
    self.solvers = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"] 
    self.penalties = {
      "lbfgs": ["l2", None],
      "liblinear": ["l1", "l2"],
      "newton-cg": ["l2", None],
      "newton-cholesky": ["l2", None],
      "sag": ["l2", None],
      "saga": ["elasticnet", "l1", "l2", None] 
    }
    self.c_values = [0.01, 0.1, 1]

    # extra data for sklearn LogisticRegression()
    self.l1_ratio = 0.5
    self.class_weight = "balanced"
    self.max_iter = 10000
  
  def _get_lr(self, solver, penalty, c) -> LogisticRegression:
    match penalty:
      case "elasticnet":
        return LogisticRegression(solver=solver, penalty=penalty, C=c, l1_ratio=self.l1_ratio, class_weight=self.class_weight,max_iter=self.max_iter)
      case None:
        return LogisticRegression(solver=solver, penalty=penalty, class_weight=self.class_weight, max_iter=self.max_iter)
      case _:
        return LogisticRegression(solver=solver, penalty=penalty, C=c, class_weight=self.class_weight, max_iter=self.max_iter)

  def _optimize(self, y_train, y_val) -> dict:
    data = {
      "score": -1,
      "solver": "",
      "penalty": "",
      "c": 1
    }

    for solver in self.solvers:
      for penalty in self.penalties[solver]:
        for c in self.c_values:
          logRegr = self._get_lr(solver, penalty, c)
          logRegr.fit(self.x_train_scaled, y_train)
          predictions = logRegr.predict(self.x_val_scaled)

          f1_score = f1_score(y_val, predictions, average="macro")
          if (f1_score > data["score"]):
            data["score"] = f1_score
            data["solver"] = solver
            data["penalty"] = penalty
            data["c"] = c
    
    return data
  
  def get_optimized_label(self) -> dict:
    return (self._optimize(self.label_train, self.label_val))

  def get_optimized_attack(self) -> dict:
    return (self._optimize(self.attack_train, self.attack_val))
  
  def print_label_report(self, model: LogisticRegression) -> None:
    label_predictions = model.predict(self.x_test_scaled)
    print("\nClassifier: Logistic Regression\n" + classification_report(self.label_test, label_predictions))
  
  def print_attack_report(self, model: LogisticRegression) -> None:
    attack_predictions = model.predict(self.x_test_scaled)
    print("\nClassifier: Logistic Regression\n" + classification_report(self.attack_test, attack_predictions, labels=ATTACK_CAT_STR_VALUES))

  def get_trained_label(self, solver="lbfgs", penalty="l2", c=1) -> LogisticRegression:
    logRegr = self._get_lr(solver, penalty, c)
    logRegr.fit(self.x_train_scaled, self.label_train)
    return logRegr

  def get_trained_attack(self, solver="newton-cg", penalty=None, c=0.01) -> LogisticRegression:
    logRegr = self._get_lr(solver, penalty, c)
    logRegr.fit(self.x_train_scaled, self.attack_train)
    return logRegr
  
  def pickle_save_label(self) -> None:
    with open("LogisticRegressionLabelModel.pkl", "wb") as file:
      pickle.dump(self.get_trained_label(), file)

  def pickle_save_attack(self) -> None:
    with open("LogisticRegressionAttackModel.pkl", "wb") as file:
      pickle.dump(self.get_trained_attack(), file)

  def pickle_load_label(self) -> LogisticRegression:
    with open("LogisticRegressionLabelModel.pkl", "rb") as file:
      data = pickle.load(file)
    return data

  def pickle_load_attack(self) -> LogisticRegression:
    with open("LogisticRegressionAttackModel.pkl", "rb") as file:
      data = pickle.load(file)
    return data

if __name__ == '__main__':
  # load data
  os.chdir(os.path.dirname(os.path.abspath(__file__)))
  df = pd.read_csv("UNSW-NB15-BALANCED-TRAIN.csv", skipinitialspace=True, low_memory=False)
  df = df.replace(r'\s+', '', regex=True)
  df.fillna('None', inplace=True)

  feature_names = []
  y_attack = df["attack_cat"]
  y_label = df["Label"]
  df = df.drop(["attack_cat", "Label"], axis=1)

  for idx, x in enumerate(df.dtypes):
    if df.dtypes.iloc[idx] == object:
      df[df.dtypes.index[idx]].astype('str')
      feature_names.append(df.dtypes.index[idx])

  df["sport"] = pd.to_numeric(df["sport"], errors="coerce")
  df["dsport"] = pd.to_numeric(df["dsport"], errors="coerce")
  df[feature_names] = df[feature_names].apply(lambda x: pd.factorize(x)[0])

  #split into train, validation and test sets
  df_label = df[SELECTED_FEATURES_LABEL_RFE]
  df_attack = df[SELECTED_FEATURES_ATTACK_CAT_RFE]
  x_train, x_temp, label_train, label_temp, attack_train, attack_temp = train_test_split(df_attack, y_label, y_attack, test_size=0.2, random_state=1)
  x_val, x_test, label_val, label_test, attack_val, attack_test =  train_test_split(x_temp, label_temp, attack_temp, test_size=0.5, random_state=1)

  lr = LogisticRegressionClassifier(x_train, x_val, x_test, label_train, label_val, label_test, attack_train, attack_val, attack_test)

  data1 = lr.pickle_load_label()
  lr.print_label_report(data1)