import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import time

# load data
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv("UNSW-NB15-BALANCED-TRAIN.csv", skipinitialspace=True, low_memory=False, keep_default_na=False)
df = df.replace(r'\s+', '', regex=True)
df.replace({'attack_cat': {'Backdoor':'Backdoors'}}, inplace=True)
ATTACK_CAT_STR_VALUES = ['None', 'Generic', 'Fuzzers', 'Exploits', 'DoS', 'Reconnaissance', 'Backdoors', 'Analysis', 'Shellcode', 'Worms']

# rfe
label_features = ['srcip', 'dstip', 'dsport', 'proto', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'Spkts', 'swin', 'stcpb', 'smeansz', 'dmeansz', 'res_bdy_len', 'Djit', 'Stime', 'Ltime', 'Dintpkt', 'tcprtt', 'synack', 'ct_state_ttl', 'ct_flw_http_mthd', 'ct_srv_src', 'ct_srv_dst', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']

# decision tree
attack_features = ['Sload', 'sbytes', 'smeansz', 'sport', 'dur', 'dstip', 'Ltime', 'Dload', 'Stime', 'Dintpkt', 'ct_state_ttl', 'sttl', 'srcip', 'dsport', 'Sintpkt', 'dbytes']

feature_names = []
y_attack = pd.factorize(df["attack_cat"])[0]
y_label = df["Label"]
df = df.drop(["attack_cat", "Label"], axis=1)

for idx, x in enumerate(df.dtypes):
  if df.dtypes.iloc[idx] == object:
    df[df.dtypes.index[idx]].astype('str')
    feature_names.append(df.dtypes.index[idx])

df["sport"] = pd.to_numeric(df["sport"], errors="coerce")
df["dsport"] = pd.to_numeric(df["dsport"], errors="coerce")
df[feature_names] = df[feature_names].apply(lambda x: pd.factorize(x)[0])

# split into train, validation and test sets
df_label = df[label_features]
df_attack = df[label_features] # switched to rfe for atk cat aswell
x_train, x_temp, label_train, label_temp, attack_train, attack_temp = train_test_split(df_attack, y_label, y_attack, test_size=0.3, random_state=1)
x_val, x_test, label_val, label_test, attack_val, attack_test =  train_test_split(x_temp, label_temp, attack_temp, test_size=0.5, random_state=1)

# train data 
solvers = ["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]
penalties = {
  "lbfgs": ["l2", None],
  "liblinear": ["l1", "l2"],
  "newton-cg": ["l2", None],
  "newton-cholesky": ["l2", None],
  "sag": ["l2", None],
  "saga": ["elasticnet", "l1", "l2", None]
}
c_values = [0.01, 0.1, 1]

top_score = 0
top_solver = ""
top_penalty = ""
top_c = 0
clf_report = dict()

scalar = StandardScaler()
x_train_scaled = scalar.fit_transform(x_train)
x_val_scaled = scalar.fit_transform(x_val)

for solver in solvers:
  for penalty in penalties[solver]:
    for c in c_values:
      logRegr = LogisticRegression(solver=solver, penalty=penalty, C=c, l1_ratio=0.5, max_iter=8085)
      logRegr.fit(x_train_scaled, attack_train)
      attack_predictions = logRegr.predict(x_val_scaled)

      acc_score = accuracy_score(attack_val, attack_predictions)
      print("====================\n" + solver + "(pen: " + str(penalty) + ", C: " + str(c) + "): " + str(acc_score) + "\n=======================")
      if (acc_score > top_score):
        top_score = acc_score
        top_solver = solver
        top_penalty = penalty
        clf_report = classification_report(attack_val, attack_predictions, target_names=ATTACK_CAT_STR_VALUES)

print("\nClassifier: Logistic Regression")
print(top_solver + "(pen: " + str(top_penalty) + ", C: " + str(top_c) + "): " + str(top_score))
print(clf_report)

