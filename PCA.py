import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# load data
df = pd.read_csv("D:/Workspace/COMP_8085_AI/COMP8085_PROJECT1_NIDS/UNSW-NB15-BALANCED-TRAIN.csv", low_memory=False, keep_default_na=False)
names = []

df = df.drop(["attack_cat", "Label"], axis=1)

for idx, x in enumerate(df.dtypes):
  if df.dtypes.iloc[idx] == object:
    df[df.dtypes.index[idx]].astype('str')
    names.append(df.dtypes.index[idx])

df["sport"] = pd.to_numeric(df["sport"], errors="coerce")
df["dsport"] = pd.to_numeric(df["dsport"], errors="coerce")
df[names] = df[names].apply(lambda x: pd.factorize(x)[0])

# analyze data
pca = PCA(n_components=3)
fit = pca.fit(df)

loadings = pd.DataFrame(abs(pca.components_), columns=df.columns).to_dict(orient="records")
pc1_loadings = dict(sorted(loadings[0].items(), key=lambda item: item[1], reverse=True))
pc2_loadings = dict(sorted(loadings[1].items(), key=lambda item: item[1], reverse=True))
pc3_loadings = dict(sorted(loadings[2].items(), key=lambda item: item[1], reverse=True))

# print analyzed data 
print("Explained Variance: %s" % fit.explained_variance_ratio_)
print(pc3_loadings)

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