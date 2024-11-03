from ucimlrepo import fetch_ucirepo

# fetch dataset
covertype = fetch_ucirepo(id=31)

# data (as pandas dataframes)
X = covertype.data.features
y = covertype.data.targets

# metadata
print(covertype.metadata)

# variable information
print(covertype.variables)


import umap
from umap.parametric_umap import ParametricUMAP
from sklearn.preprocessing import StandardScaler

#
# # Pré-traitements
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(strategy="mean")  # Choisir la stratégie (mean, median, most_frequent)
# X = imputer.fit_transform(X)
#
# #standardiser données
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)


# reducer = umap.UMAP(n_neighbors=15, min_dist=0.1)
# embedding = reducer.fit(X_scaled)

# ### version paramétrique
# embedder = ParametricUMAP()
# embedding = embedder.fit_transform(X)
