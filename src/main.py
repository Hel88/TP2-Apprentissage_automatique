from ucimlrepo import fetch_ucirepo
import umap
from umap.parametric_umap import ParametricUMAP
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# fetch dataset
print("fetch data")
covertype = fetch_ucirepo(id=31)
print("done fetching data")


# data (as pandas dataframes)
X = covertype.data.features
y = covertype.data.targets

# metadata
print(covertype.metadata)

# variable information
print(covertype.variables)

# Sélectionner seulement 10% de la base de données au hasard
X_sample, _, y_sample, _ = train_test_split(X, y, train_size=0.1, stratify=y, random_state=42)
print(X_sample.shape)
print(y_sample.shape)


#
# # # Pré-traitements
# # from sklearn.impute import SimpleImputer
# # imputer = SimpleImputer(strategy="mean")  # Choisir la stratégie (mean, median, most_frequent)
# # X = imputer.fit_transform(X)
#
# #standardiser données
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
#
# print("Lancement de UMAP...")
# reducer = umap.UMAP()
# embedding = reducer.fit_transform(X_scaled)
# print("Umap fini")
# print(embedding.shape)
#
#
# # ### version paramétrique
# # embedder = ParametricUMAP()
# # embedding = embedder.fit_transform(X)
