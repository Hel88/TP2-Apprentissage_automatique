from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report
from ucimlrepo import fetch_ucirepo
from tqdm import tqdm
import time

# Initialisation des étapes
steps = [
    "Fetching dataset",
    "Reducing dataset",
    "Splitting train/test data",
    "Creating pipeline",
    "Training pipeline",
    "Making predictions",
    "Evaluating model",
]
progress = tqdm(steps)

# Récupérer le dataset
progress.set_description("Fetching dataset...")
covertype = fetch_ucirepo(id=31)
time.sleep(0.5)  # Simuler une pause pour visualisation
progress.update()

# Données
X = covertype.data.features
y = covertype.data.targets

# Afficher la taille et la distribution
progress.set_description("Reducing dataset...")
print("Taille des données :", X.shape, y.shape)
print("Distribution des classes :\n", y.Cover_Type.value_counts())

# Réduire le dataset
new_size = 0.7  # Pourcentage du dataset à garder
y = y.values.ravel()  # Convertir en tableau unidimensionnel
X_sampled, _, y_sampled, _ = train_test_split(X, y, train_size=new_size, stratify=y, random_state=42)
time.sleep(0.5)
progress.update()

# Diviser les données en ensemble d'entraînement et de test
progress.set_description("Splitting train/test data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_sampled, y_sampled, test_size=0.3, stratify=y_sampled, random_state=42
)
time.sleep(0.5)
progress.update()


progress.set_description("Creating pipeline...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])
time.sleep(0.5)
progress.update()

# Entraîner le pipeline
progress.set_description("Training pipeline...")
pipeline.fit(X_train, y_train)
time.sleep(0.5)
progress.update()

# Prédire sur l'ensemble de test
progress.set_description("Making predictions...")
y_pred = pipeline.predict(X_test)
time.sleep(0.5)
progress.update()

# Évaluer les performances
progress.set_description("Evaluating model...")
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average="weighted")
progress.update()
progress.close()  # Fermer la barre de progression

# Afficher les résultats finaux
print("Résultats :")
print(f"Accuracy : {accuracy:.4f}")
print(f"F1-Score : {f1:.4f}")
print("\nRapport de classification détaillé :\n")
print(classification_report(y_test, y_pred))
