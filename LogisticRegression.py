from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from Constants import *
import pandas as pd
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

    self.scalar = StandardScaler()
    #self.x_train_label_scaled = self.scalar.fit_transform(self.x_train[SELECTED_FEATURES_LABEL_RFE])
    #self.x_train_attack_scaled = self.scalar.fit_transform(self.x_train[SELECTED_FEATURES_ATTACK_CAT_RFE])

    # self.x_val_label_scaled = self.scalar.transform(self.x_val[SELECTED_FEATURES_LABEL_RFE])
    # self.x_val_attack_scaled = self.scalar.transform(self.x_val[SELECTED_FEATURES_ATTACK_CAT_RFE])

    # self.x_test_label_scaled = self.scalar.transform(self.x_test)
    # self.x_test_attack_scaled = self.scalar.transform(self.x_test)

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
    self.l1_ratio_values = [0.25, 0.5, 0.75]

    # extra data for sklearn LogisticRegression()
    self.class_weight = "balanced"
    self.max_iter = 10000
  
  def _get_lr(self, solver, penalty, c, ratio=None) -> LogisticRegression:
    match penalty:
      case "elasticnet":
        return LogisticRegression(solver=solver, penalty=penalty, C=c, l1_ratio=ratio, class_weight=self.class_weight,max_iter=self.max_iter)
      case None:
        return LogisticRegression(solver=solver, penalty=penalty, class_weight=self.class_weight, max_iter=self.max_iter)
      case _:
        return LogisticRegression(solver=solver, penalty=penalty, C=c, class_weight=self.class_weight, max_iter=self.max_iter)

  def _optimize(self, x_scaled_train, y_train, x_scaled_val, y_val) -> dict:
    data = {
      "score": -1,
      "solver": "",
      "penalty": "",
      "c": 1,
      "l1_ratio": None,
      "time": float("inf")
    }

    def _update_score(l1_ratio=None) -> None:
      start_time = time.time()
      logRegr = self._get_lr(solver, penalty, c, l1_ratio)
      logRegr.fit(x_scaled_train, y_train) 
      predictions = logRegr.predict(x_scaled_val) 
      score = f1_score(y_val, predictions, average="macro")
      time_taken = time.time() - start_time
      print(score, end=", ")
      print(time_taken)
      if (score > data["score"] or score == data["score"] and time_taken < data["time"]):
        data["score"] = score
        data["solver"] = solver
        data["penalty"] = penalty
        data["c"] = c
        data["l1_ratio"] = l1_ratio
        data["time"] = time_taken

    for solver in self.solvers:
      for penalty in self.penalties[solver]:
        for c in self.c_values:
          if penalty == "elasticnet":
            for ratio in self.l1_ratio_values:
              _update_score(ratio)
          else:
            _update_score()
    return data
  
  def get_optimized_label_settings(self) -> dict:
    x_scaled_train = self.scalar.fit_transform(x_train[SELECTED_FEATURES_LABEL_RFE])
    x_scaled_val = self.scalar.transform(x_val[SELECTED_FEATURES_LABEL_RFE])
    return (self._optimize(x_scaled_train, self.label_train, x_scaled_val, self.label_val))

  def get_optimized_attack_settings(self) -> dict:
    x_scaled_train = self.scalar.fit_transform(x_train[SELECTED_FEATURES_ATTACK_CAT_RFE])
    x_scaled_val = self.scalar.transform(x_val[SELECTED_FEATURES_ATTACK_CAT_RFE])
    return (self._optimize(x_scaled_train, self.attack_train, x_scaled_val, self.attack_val))
  
  def print_label_report(self, model: LogisticRegression) -> None:
    self.scalar.fit_transform(x_train[SELECTED_FEATURES_LABEL_RFE]) # ???
    x_scaled_test = self.scalar.transform(x_val[SELECTED_FEATURES_LABEL_RFE])
    label_predictions = model.predict(x_scaled_test)
    print("\nClassifier: Logistic Regression\n" + classification_report(self.label_test, label_predictions))
  
  def print_attack_report(self, model: LogisticRegression) -> None:
    self.scalar.fit_transform(x_train[SELECTED_FEATURES_ATTACK_CAT_RFE]) # ???
    x_scaled_test = self.scalar.transform(x_val[SELECTED_FEATURES_ATTACK_CAT_RFE])
    attack_predictions = model.predict(x_scaled_test)
    print("\nClassifier: Logistic Regression\n" + classification_report(self.attack_test, attack_predictions, labels=ATTACK_CAT_STR_VALUES))

  def get_trained_label(self, solver="lbfgs", penalty="l2", c=1, ratio=None) -> LogisticRegression:
    logRegr = self._get_lr(solver, penalty, c, ratio)
    x_scaled_train = self.scalar.fit_transform(x_train[SELECTED_FEATURES_LABEL_RFE])
    logRegr.fit(x_scaled_train, self.label_train)
    return logRegr

  def get_trained_attack(self, solver="newton-cg", penalty=None, c=0.01, ratio=None) -> LogisticRegression:
    logRegr = self._get_lr(solver, penalty, c, ratio)
    x_scaled_train = self.scalar.fit_transform(x_train[SELECTED_FEATURES_ATTACK_CAT_RFE])
    logRegr.fit(x_scaled_train, self.attack_train)
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
  x_train, x_temp, label_train, label_temp, attack_train, attack_temp = train_test_split(df, y_label, y_attack, test_size=0.2, random_state=1)
  x_val, x_test, label_val, label_test, attack_val, attack_test =  train_test_split(x_temp, label_temp, attack_temp, test_size=0.5, random_state=1)

  lr = LogisticRegressionClassifier(x_train, x_val, x_test, label_train, label_val, label_test, attack_train, attack_val, attack_test)

  opt_label_dict = lr.get_optimized_label_settings()
  print(opt_label_dict)

  opt_atk_dict = lr.get_optimized_attack_settings()
  print(opt_atk_dict)
  
  #lr.pickle_save_label()
  #hi = lr.pickle_load_label()
  #label_data = lr.get_trained_label(boop["solver"], boop["penalty"], boop["c"], boop["l1_ratio"])
  #lr.print_label_report(hi)

  #print(lr.get_optimized_attack_settings())
  #lr.print_attack_report(lr.get_trained_attack())


  #   x_train, x_temp, label_train, label_temp, attack_train, attack_temp = train_test_split(df, y_label, y_attack, test_size=0.2, random_state=1)
  # x_val, x_test, label_val, label_test, attack_val, attack_test =  train_test_split(x_temp, label_temp, attack_temp, test_size=0.5, random_state=1)

  #x_train1, x_remove, label_train1, label_remove, attack_train1, attack_remove = train_test_split(df, y_label, y_attack, test_size=0.9, random_state=1)
  #x_train, x_temp, label_train, label_temp, attack_train, attack_temp = train_test_split(x_train1, label_train1, attack_train1, test_size=0.2, random_state=1)
  #x_val, x_test, label_val, label_test, attack_val, attack_test =  train_test_split(x_temp, label_temp, attack_temp, test_size=0.5, random_state=1)