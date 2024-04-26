import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle
import datetime
import time

# Début du comptage du temps
start_time = time.time()

# Chargement des données sans en-tête
data_path = './DATA/IN/extract_gold_dvf_11_04_24_true_gold.csv'
data = pd.read_csv(data_path, header=None, sep=';')

print(f"Original data shape: {data.shape}")  # Imprimer la forme originale des données
data.drop(data.columns[[6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17]], axis=1, inplace=True)
print(f"Data shape after column drop: {data.shape}")  # Imprimer la forme des données après la suppression

# Extraire l'année de la colonne index 1
data[1] = pd.to_datetime(data[1]).dt.year
print("date extraite")

# Remplacement des virgules par des points et conversion en float pour les colonnes index 0, 2, 3, 4
for col in [0, 2, 3, 4]:
    data[col] = data[col].astype(str).str.replace(',', '.').astype(float)
print("Conversion des , en . faites")

# Définition des colonnes numériques et catégorielles par leurs indices
numeric_features = [0, 2, 3, 4]  # indices des colonnes numériques
categorical_features = [5, 6]  # indices des deux dernières colonnes

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
# Configuration du préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])
print("fin du Préprocesseur")

print("début du pipeline complet avec RandomForestRegressor")
# Configuration du Gradient Boosting Regressor
gbm_model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('regressor', GradientBoostingRegressor())])
print("fin du pipeline complet avec RandomForestRegressor")

# Séparation des données
X = data.drop(0, axis=1)  # Suppression de la colonne index 0 (price)
y = data[0]  # Utilisation de la colonne index 0 comme cible

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("début d'Entraînement du modèle'")
# Entraînement du modèle GBM
gbm_model.fit(X_train, y_train)
print("Fin d'Entraînement du modèle")

param_grid = {
    'regressor__n_estimators': [200, 300, 400],
    'regressor__learning_rate': [0.1, 0.15, 0.2, 0.25],
    'regressor__max_depth': [7, 9, 11],
    'regressor__min_samples_split': [2, 4, 6],
    'regressor__min_samples_leaf': [1, 2, 3]
}

# Création du GridSearchCV
grid_search = GridSearchCV(gbm_model, param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
grid_search.fit(X_train, y_train)

# Affichage des meilleurs paramètres et score
print("Best parameters:", grid_search.best_params_)
print("Best MSE:", -grid_search.best_score_)

# Utiliser le meilleur modèle
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
best_mse = mean_squared_error(y_test, y_pred_best)
best_mae = mean_absolute_error(y_test, y_pred_best)
best_r2 = r2_score(y_test, y_pred_best)

print("Best Model Performance Metrics:")
print(f"Mean Squared Error: {best_mse}")
print(f"Mean Absolute Error: {best_mae}")
print(f"R-squared: {best_r2}")

# Fin du comptage du temps
end_time = time.time()
training_duration_seconds = end_time - start_time
minutes = int(training_duration_seconds // 60)
seconds = int(training_duration_seconds % 60)
training_duration_str = f"{minutes} minutes and {seconds} seconds"

# Évaluation du modèle GBM
y_pred_gbm = gbm_model.predict(X_test)
mse_gbm = mean_squared_error(y_test, y_pred_gbm)
mae_gbm = mean_absolute_error(y_test, y_pred_gbm)
r2_gbm = r2_score(y_test, y_pred_gbm)

metrics = {
    "Mean Squared Error": mse_gbm,
    "Mean Absolute Error": mae_gbm,
    "R-squared": r2_gbm
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
log_file_path = f'./MODELS/LOG/training_log_{file_datetime_str}.txt'

# Écriture des informations dans un fichier texte
with open(log_file_path, 'w') as file:
    file.write(f"Training Date and Time: {datetime_str}\n")
    file.write(f"Model Trained: GradientBoostingRegressor\n")
    file.write(f"Data Trained On: {len(y_train)} samples\n")
    file.write(metrics_str)
    file.write(f"Training Duration: {training_duration_str}\n\n")

# Sauvegarde du modèle
model_path = f'./MODELS/GradientBoostingRegressor{file_datetime_str}.pkl'
with open(model_path, 'wb') as file:
    pickle.dump(gbm_model, file)
