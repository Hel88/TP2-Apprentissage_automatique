from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from UMAP_module import UMAP_module


# Récupérer le dataset
print("Fetching dataset...")
covertype = fetch_ucirepo(id=31)
print("done")

# data
X = covertype.data.features
y = covertype.data.targets

print(X.shape)
print(y.shape)

print(y.Cover_Type.value_counts())


# Réduire le dataset
new_size = 0.2   # pourcentage du dataset qu'on veut garder
X_sampled, _, y_sampled, _ = train_test_split(X, y, train_size=new_size, stratify=y, random_state=42)

# Standardiser données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sampled)

# UMAP visualisation
umap = UMAP_module(X_scaled, y_sampled)
umap.run_algo()
umap.display("UMAP plotting with 50% of the dataset", "umap_20%_of_dataset")