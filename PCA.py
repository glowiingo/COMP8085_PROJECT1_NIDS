import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import time
import os

# load data
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv("UNSW-NB15-BALANCED-TRAIN.csv", skipinitialspace=True, low_memory=False, keep_default_na=False)
df = df.replace(r'\s+', '', regex=True)
df.replace({'attack_cat': {'Backdoor':'Backdoors'}}, inplace=True)

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

x_train, x_test, y_train, y_test = train_test_split(df, y_attack, test_size=0.3, random_state=1)

# standardize data
scalar = StandardScaler()
scalar.fit(x_train) 
x_train_scalar = scalar.transform(x_train)
x_test_scalar = scalar.transform(x_test)

# analyze data with PCA and LogisticRegression
start = time.perf_counter()
pca = PCA()
fit = pca.fit(x_train_scalar) 

loadings = pd.DataFrame(abs(pca.components_), columns=df.columns).to_dict(orient="records")

# print explained variance
# print("Explained Variance: %s" % fit.explained_variance_ratio_)

feature_set = set()
score = 0
score_names = []

for i in range(len(fit.explained_variance_ratio_)):
  names = []
  for item in sorted(loadings[i].items(), key=lambda item: item[1], reverse=True):
    if (item[1] > 0.1):
      names.append(item[0])
    else:
      break
  
  x_train_selected= x_train[names]  
  x_test_selected = x_test[names]

  scalar.fit(x_train_selected) 
  x_train_scalar_trans = scalar.transform(x_train_selected)
  x_test_scalar_trans = scalar.transform(x_test_selected)
  pca.fit(x_train_scalar_trans)

  x_train_trans = pca.transform(x_train_scalar_trans)
  x_test_trans = pca.transform(x_test_scalar_trans)

  logRegr = LogisticRegression(solver="lbfgs", max_iter=1500) 
  logRegr.fit(x_train_trans, y_train)

  logRegr.predict(x_test_trans[0:len(y_test)])
  if (logRegr.score(x_test_trans, y_test) > score):
    score = logRegr.score(x_test_trans, y_test)
    score_names = names 

end = time.perf_counter()
exp_time = end - start

print(exp_time)
print(score)
print(score_names)

# Label
# 43.70877190004103 seconds
# 0.9920037943070573
# ['state', 'ct_dst_sport_ltm', 'sttl', 'dttl', 'dwin', 'swin', 'is_sm_ips_ports', 'Sintpkt', 'ct_state_ttl', 'ct_src_ ltm', 'Sload']
#
# Attack Cat
# 887.9623371999478 seconds
# 0.9178740023269774
# ['ct_dst_src_ltm', 'ct_dst_sport_ltm', 'ct_srv_dst', 'ct_src_dport_ltm', 'ct_srv_src', 'dwin', 'swin', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_state_ttl', 'sttl', 'state', 'service', 'sport', 'stcpb', 'dtcpb', 'dmeansz', 'is_ftp_login', 'Stime', 'Ltime', 'ct_ftp_cmd', 'dttl', 'Dload', 'srcip', 'Sload', 'dsport']