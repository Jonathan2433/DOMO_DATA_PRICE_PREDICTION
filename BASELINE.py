import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pickle
import datetime
import time

# Début du comptage du temps
start_time = time.time()

# Chargement des données sans en-tête
data_path = './DATA/IN/DATA_GOLD_FOR_ML.csv'
data = pd.read_csv(data_path, sep=';', low_memory=False)

print(f"Original data shape: {data.shape}")  # Imprimer la forme originale des données

# Afficher les noms des colonnes
print("Columns:", data.columns)

# Remplacement des virgules par des points et conversion en float pour les colonnes numériques
numeric_cols = ['Prix', 'SurfaceTerrain', 'SurfaceBati', 'SurfaceCarrez']
for col in numeric_cols:
    data[col] = data[col].astype(str).str.replace(',', '.').astype(float)
print("Conversion des , en . faites")

# Définition des colonnes numériques et catégorielles par leurs noms
numeric_features = ['SurfaceTerrain', 'SurfaceBati', 'SurfaceCarrez', 'NombrePiecesPrincipales']
categorical_features = ['DT_Annee', 'CodePostal', 'TypeLocalName']

print("début de la Pipeline de prétraitement pour les caractéristiques numériques")
# Pipeline de prétraitement pour les caractéristiques numériques
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())])
print("Fin de la Pipeline de prétraitement pour les caractéristiques numériques")

print("début de la Pipeline de prétraitement pour les caractéristiques catégorielles")
# Pipeline de prétraitement pour les caractéristiques catégorielles
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
print("fin de la Pipeline de prétraitement pour les caractéristiques catégorielles")

print("début du Préprocesseur")
# Préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
print("fin du Préprocesseur")

print("début du pipeline complet avec RandomForestRegressor")
# Création du pipeline complet avec RandomForestRegressor
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', RandomForestRegressor())])
print("fin du pipeline complet avec RandomForestRegressor")

# Séparation des données
X = data.drop('Prix', axis=1)  # Suppression de la colonne 'Prix'
y = data['Prix']  # Utilisation de la colonne 'Prix' comme cible

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("début d'Entraînement du modèle'")
# Entraînement du modèle
model.fit(X_train, y_train)
print("Fin d'Entraînement du modèle")

# Fin du comptage du temps
end_time = time.time()
training_duration_seconds = end_time - start_time
minutes = int(training_duration_seconds // 60)
seconds = int(training_duration_seconds % 60)
training_duration_str = f"{minutes} minutes and {seconds} seconds"

# Évaluation du modèle
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {
    "Mean Squared Error": mse,
    "Mean Absolute Error": mae,
    "R-squared": r2
}

# Affichage et enregistrement des métriques
print("Model Performance Metrics:")
metrics_str = ""
for metric, value in metrics.items():
    print(f"{metric}: {value}")
    metrics_str += f"{metric}: {value}\n"

print(f"Training Duration: {training_duration_str}")

# Date et heure actuelles
current_datetime = datetime.datetime.now()
datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
file_datetime_str = current_datetime.strftime("%Y%m%d_%H%M%S")  # Pour nom de fichier unique

# Chemin du fichier de log unique
log_file_path = f'./MODELS/BASELINE/LOG/training_log_{file_datetime_str}.txt'

# Écriture des informations dans un fichier texte
with open(log_file_path, 'w') as file:
    file.write(f"Training Date and Time: {datetime_str}\n")
    file.write(f"Model Trained: RandomForestRegressor\n")
    file.write(f"Data Trained On: {len(y_train)} samples\n")
    file.write(metrics_str)
    file.write(f"Training Duration: {training_duration_str}\n\n")

# Sauvegarde du modèle
model_path = f'./MODELS/BASELINE/RF_REGRESSOR_BASELINE_{file_datetime_str}.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(model, file)