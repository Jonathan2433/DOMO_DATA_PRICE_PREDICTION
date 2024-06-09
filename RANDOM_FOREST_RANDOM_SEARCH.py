import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import datetime
import time
import pickle

def print_time(message, start_time):
    elapsed_time = time.time() - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    print(f"{message} - {minutes} minutes and {seconds} seconds elapsed.")

# Début du comptage du temps
start_time = time.time()
print("Starting script...")

# Chargement des données sans en-tête
data_path = './DATA/IN/DATA_GOLD_FOR_ML.csv'
data = pd.read_csv(data_path, sep=';', low_memory=False)
print(f"Original data shape: {data.shape}")

# Conversion des colonnes numériques
print_time("Converting numeric columns", start_time)
numeric_cols = ['Prix', 'SurfaceTerrain', 'SurfaceBati', 'SurfaceCarrez', 'NombrePiecesPrincipales']
for col in numeric_cols:
    data[col] = data[col].astype(str).str.replace(',', '.').astype(float)

# Exclusion des outliers
print_time("Excluding outliers", start_time)
for col in ['Prix', 'SurfaceTerrain', 'SurfaceBati', 'SurfaceCarrez', 'NombrePiecesPrincipales']:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
print(f"Data shape after outlier removal: {data.shape}")

# Configuration du préprocesseur
print_time("Setting up preprocessor", start_time)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())]), ['SurfaceTerrain', 'SurfaceBati', 'SurfaceCarrez', 'NombrePiecesPrincipales']),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), ['DT_Annee', 'CodePostal', 'TypeLocalName'])
    ])

# Pipeline du modèle
print_time("Setting up pipeline", start_time)
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Séparation des données
print_time("Splitting data", start_time)
X = data.drop('Prix', axis=1)  # Suppression de la colonne 'Prix'
y = data['Prix']  # Utilisation de la colonne 'Prix' comme cible
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")

# Paramètres pour RandomizedSearchCV
print_time("Setting up RandomizedSearchCV parameters", start_time)
param_dist = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [10, 20, 30],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2],
    'regressor__max_features': ['sqrt', 'log2', None]
}

# Exécution du RandomizedSearchCV
print_time("Starting RandomizedSearchCV", start_time)
random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=50, cv=3, verbose=1, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Résultats
best_params = random_search.best_params_
print(f"Best parameters: {best_params}")
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Calcul du temps d'exécution
end_time = time.time()
training_duration_seconds = end_time - start_time
minutes = int(training_duration_seconds // 60)
seconds = int(training_duration_seconds % 60)
training_duration_str = f"{minutes} minutes and {seconds} seconds"

# Enregistrement des résultats et des paramètres dans le fichier de log
current_datetime = datetime.datetime.now()
datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
file_datetime_str = current_datetime.strftime("%Y%m%d_%H%M%S")
log_file_path = f'./MODELS/LOG/training_log_{file_datetime_str}.txt'

model_path = f'./MODELS/RF_RANDOM_SEARCH_{file_datetime_str}.pkl'
with open(log_file_path, 'w') as file:
    file.write(f"Training Date and Time: {datetime_str}\n")
    file.write(f"Model Trained: RandomForestRegressor RANDOM SEARCH\n")
    file.write(f"Data Trained On: {len(y_train)} samples\n")
    file.write(f"Best Parameters: {best_params}\n")
    file.write(f"Mean Squared Error: {mse}\n")
    file.write(f"Mean Absolute Error: {mae}\n")
    file.write(f"R-squared: {r2}\n")
    file.write(f"Training Duration: {training_duration_str}\n\n")

with open(model_path, 'wb') as file:
    pickle.dump(best_model, file)

print_time("Script completed", start_time)