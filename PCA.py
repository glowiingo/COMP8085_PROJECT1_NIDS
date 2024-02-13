import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np
import os


# load data
os.chdir(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv("UNSW-NB15-BALANCED-TRAIN.csv", low_memory=False, keep_default_na=False)
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

x_train, x_test, y_train, y_test = train_test_split(df, y_label, test_size=0.3, random_state=1)

# standardize data
scalar = StandardScaler()
scalar.fit(x_train) 
x_train_scalar = scalar.transform(x_train)
x_test_scalar = scalar.transform(x_test)

# analyze data with PCA and LogisticRegression
pca = PCA()
fit = pca.fit(x_train_scalar) # fit = pca.fit(x_train)
# fit = pca.fit(x_train)
# x_train_trans = pca.transform(x_train_scalar)
# x_test_trans = pca.transform(x_test_scalar)

#logRegr = LogisticRegression(solver="lbfgs")
#logRegr.fit(x_train, y_train)

#logRegr.predict(x_test[0:10])
#print(logRegr.score(x_test, y_test))


loadings = pd.DataFrame(abs(pca.components_), columns=df.columns).to_dict(orient="records")

# print analyzed data 
#print("Explained Variance: %s" % fit.explained_variance_ratio_)

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
  #print(names)
  x_train_selected= x_train[names]  
  x_test_selected = x_test[names]

  scalar.fit(x_train_selected) 
  x_train_scalar_trans = scalar.transform(x_train_selected)
  x_test_scalar_trans = scalar.transform(x_test_selected)
  pca.fit(x_train_scalar_trans)

  x_train_trans = pca.transform(x_train_scalar_trans)
  x_test_trans = pca.transform(x_test_scalar_trans)

  logRegr = LogisticRegression(solver="lbfgs", max_iter=250)
  logRegr.fit(x_train_trans, y_train)

  logRegr.predict(x_test_trans[0:10])

  if (logRegr.score(x_test_trans, y_test) > score):
    score = logRegr.score(x_test_trans, y_test)
    score_names = names 
  #print(logRegr.score(x_test_trans, y_test))

print(score)
print(score_names)
"""
# Scree Plot Code From https://www.jcchouinard.com/pca-scree-plot/
plt.bar(
    range(1,len(pca.explained_variance_)+1),
    pca.explained_variance_
    )
 
plt.plot(
    range(1,len(pca.explained_variance_ )+1),
    np.cumsum(pca.explained_variance_),
    c='red',
    label='Cumulative Explained Variance')
 
plt.legend(loc='center right')
plt.xlabel('Number of components')
plt.ylabel('Explained variance (eignenvalues)')
plt.title('Scree plot')
 
plt.show()
"""