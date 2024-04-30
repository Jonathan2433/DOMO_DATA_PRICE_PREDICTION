import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pickle
import datetime
import time

# Début du comptage du temps
start_time = time.time()

# Chargement des données sans en-tête
data_path = './DATA/IN/extract_gold_dvf_11_04_24_true_gold.csv'
data = pd.read_csv(data_path, header=None, sep=';')

# Préparation des données
data.drop(data.columns[[6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17]], axis=1, inplace=True)
data[1] = pd.to_datetime(data[1]).dt.year
for col in [0, 2, 3, 4]:
    data[col] = data[col].astype(str).str.replace(',', '.').astype(float)

# Définition des colonnes numériques et catégorielles
numeric_features = [0, 1, 2, 3, 4]
categorical_features = [5, 6]

# Préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)])

# Pipeline avec Gradient Boosting Regressor
model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', GradientBoostingRegressor())])

# Séparation des données
X = data.drop(0, axis=1)
y = data[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paramètres pour RandomizedSearchCV
param_dist = {
    'regressor__n_estimators': [100, 200, 300, 400],
    'regressor__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'regressor__max_depth': [3, 5, 7, 9],
    'regressor__min_samples_split': [2, 4, 6],
    'regressor__min_samples_leaf': [1, 2, 3]
}

# Création et entraînement du RandomizedSearchCV
random_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=100, cv=3,
                                   verbose=1, random_state=42, scoring='neg_mean_squared_error')
random_search.fit(X_train, y_train)

# Résultats
best_params = random_search.best_params_
best_model = random_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Calcul des métriques pour le meilleur modèle
best_mse = mean_squared_error(y_test, y_pred_best)
best_mae = mean_absolute_error(y_test, y_pred_best)
best_r2 = r2_score(y_test, y_pred_best)

# Calcul du temps d'exécution
end_time = time.time()
training_duration_seconds = end_time - start_time
minutes = int(training_duration_seconds // 60)
seconds = int(training_duration_seconds % 60)
training_duration_str = f"{minutes} minutes and {seconds} seconds"

# Date et heure actuelles
current_datetime = datetime.datetime.now()
datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
file_datetime_str = current_datetime.strftime("%Y%m%d_%H%M%S")

# Sauvegarde des logs et du modèle
log_file_path = f'./MODELS/LOG/training_log_{file_datetime_str}.txt'
model_path = f'./MODELS/GBM_{file_datetime_str}.pkl'
with open(log_file_path, 'w') as file:
    file.write(f"Training Date and Time: {datetime_str}\n")
    file.write(f"Model Trained: GradientBoostingRegressor random search\n")
    file.write(f"Data Trained On: {len(y_train)} samples\n")
    file.write(f"Best Parameters: {best_params}\n")  # Log best parameters
    file.write(f"Mean Squared Error: {best_mse}\n")
    file.write(f"Mean Absolute Error: {best_mae}\n")
    file.write(f"R-squared: {best_r2}\n")
    file.write(f"Training Duration: {training_duration_str}\n\n")
with open(model_path, 'wb') as file:
    pickle.dump(best_model, file)

print("Best Model Performance Metrics:")
print(f"Mean Squared Error: {best_mse}")
print(f"Mean Absolute Error: {best_mae}")
print(f"R-squared: {best_r2}")
print(f"Training Duration: {training_duration_str}")
