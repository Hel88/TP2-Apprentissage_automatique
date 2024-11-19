from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from UMAP_module import UMAP_module
from imblearn.over_sampling import SMOTE



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
new_size = 0.5   # pourcentage du dataset qu'on veut garder
X_sampled, _, y_sampled, _ = train_test_split(X, y, train_size=new_size, stratify=y, random_state=42)


# Standardiser données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print ("before over-sampling")
print(y.Cover_Type.value_counts())


# Over-sampling avec SMOTE
sm = SMOTE(random_state=42, k_neighbors=3)
print("Running SMOTE over-sampling...")
X_oversampled, y_oversampled = sm.fit_resample(X_scaled, y_sampled)
print("SMOTE done\n")


print("After over-sampling: ")
print(y_oversampled.Cover_Type.value_counts())


# UMAP visualisation
umap = UMAP_module(X_oversampled, y_oversampled)
umap.run_algo()
umap.display("UMAP plotting with all of the dataset oversampled with SMOTE", "umap_SMOTE")
