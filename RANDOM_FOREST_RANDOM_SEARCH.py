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
    if col in [3, 4]:
        data[col].fillna(0, inplace=True)

# Exclusion des outliers
for col in [0, 2, 3, 4]:
    q1 = data[col].quantile(0.25)
    q3 = data[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

# Configuration du préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler())]), [0, 1, 2, 3, 4]),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), [5, 6])
    ])

# Pipeline du modèle
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Séparation des données
X = data.drop(0, axis=1)
y = data[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Paramètres pour RandomizedSearchCV
param_dist = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [10, 20, 30],
    'regressor__min_samples_split': [2, 5],
    'regressor__min_samples_leaf': [1, 2],
    'regressor__max_features': ['auto', 'sqrt']
}

# Exécution du RandomizedSearchCV
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
