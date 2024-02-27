# Feature Extraction with RFE
import matplotlib.pyplot as plt
from sklearn.base import np
from xgboost import XGBClassifier
from pandas import read_csv
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import time

# load data
file = "UNSW-NB15-BALANCED-TRAIN.csv"
col_names = ['srcip', 'sport', 'dstip',	'dsport', 'proto', 'state',	'dur', 'sbytes', 'dbytes', 'sttl',	'dttl',	'sloss', 
        'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
        'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt',	'synack', 'ackdat',
        'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',	'ct_ftp_cmd', 'ct_srv_src',	'ct_srv_dst', 
        'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'Label']

feature_cols = ['srcip', 'sport', 'dstip',	'dsport', 'proto', 'state',	'dur', 'sbytes', 'dbytes', 'sttl',	'dttl',	'sloss', 
        'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz',
        'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt',	'synack', 'ackdat',
        'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',	'ct_ftp_cmd', 'ct_srv_src',	'ct_srv_dst', 
        'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm']

dataframe = read_csv(file, names=col_names, low_memory=False)

## Dataframe Preprocessing
dataframe.replace({'attack_cat': {'Backdoor':'Backdoors'}}, inplace=True)
dataframe = dataframe.replace(r'\s+', '', regex=True)

dataframe.fillna('None', inplace=True)

categorical = dataframe[col_names]
enc = OrdinalEncoder()
enc.fit(categorical)
numerical = enc.transform(categorical)
for n, feat in enumerate(col_names):
    dataframe[feat] = numerical[:, n]

X_lbl = dataframe[feature_cols]
Y_lbl = dataframe.Label

X_lbl_train, X_lbl_test, Y_lbl_train, Y_lbl_test = train_test_split(X_lbl, Y_lbl, test_size =0.8, random_state=1)

#feature extraction
model_lbl = XGBClassifier()
rfe_lbl = RFECV(model_lbl)
start = time.time()
fit = rfe_lbl.fit(X_lbl_train, Y_lbl_train)
end = time.time()
elapsed = end - start
print("Feature selction time: " + str(elapsed))
print("Features for Label")
print("Num Features: %d" % fit.n_features_)
print("Selected Features: %s" % fit.support_)
print("Feature Ranking: %s" % fit.ranking_)
selected_ranking_lbl = np.where(rfe_lbl.ranking_ == 1)[0]
selected_cols_lbl = [col_names[i] for i in selected_ranking_lbl]

X_lbl_train_selected = X_lbl_train[selected_cols_lbl]


model_lbl.fit(X_lbl_train_selected, Y_lbl_train)
X_test_selected = X_lbl_test[selected_cols_lbl]
accuracy_lbl = model_lbl.score(X_test_selected, Y_lbl_test)
print("Accuracy on Test Set for Label: ", accuracy_lbl)



# plt.figure(figsize=(10, 6))
# plt.title("RFE - Feature Ranking for Label")
# plt.xlabel("Feature")
# plt.ylabel("Ranking")
# plt.bar(rfe_lbl.feature_names_in_, rfe_lbl.ranking_)
# plt.xticks(rotation=90)
# plt.show()

X_cat = dataframe[feature_cols]
Y_cat = dataframe.attack_cat

X_cat_train, X_cat_test,Y_cat_train, Y_cat_test = train_test_split(X_cat, Y_cat, test_size =0.3, random_state=1)
model_cat = XGBClassifier()
rfe_cat = RFECV(model_cat)
start = time.time()
fit_cat = rfe_cat.fit(X_cat_train, Y_cat_train)
end = time.time()
elapsed = end - start
print()
print("Feature selction time: " + str(elapsed))
print("Features for attack_cat")
print("Num Features: %d" % fit_cat.n_features_)
print("Selected Features: %s" % fit_cat.support_)
print("Feature Ranking: %s" % fit_cat.ranking_)


selected_ranking_cat = np.where(rfe_cat.ranking_ == 1)[0]
selected_cols_cat = [col_names[i] for i in selected_ranking_cat]
X_cat_train_selected = X_cat_train[selected_cols_cat]
X_cat_test_selected = X_cat_test[selected_cols_cat]

model_cat.fit(X_cat_train_selected, Y_cat_train)
accuracy_cat = model_cat.score(X_cat_test_selected, Y_cat_test)
print("Accuracy on Test Set for Attack Category: ", accuracy_cat)

# plt.figure(figsize=(10, 6))
# plt.title("RFE - Feature Ranking for Attack Category")
# plt.xlabel("Feature")
# plt.ylabel("Ranking")
# plt.bar(rfe_cat.feature_names_in_, rfe_cat.ranking_)
# plt.xticks(rotation=90)
# plt.show()
