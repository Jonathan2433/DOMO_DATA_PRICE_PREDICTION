import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Chemin vers le fichier du modèle sauvegardé
model_path = './MODELS/GBM_20240501_154405.pkl'

# Chargement du modèle
with open(model_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Vérification des paramètres du modèle
print("Paramètres du modèle chargé:")
print(loaded_model.get_params())

# Si vous avez accès aux données de test ici, vous pouvez également recalculer les métriques
# Chargement des données de test (exemple fictif, vous devrez utiliser votre propre chemin ou variable)
# X_test = <votre_dataframe_test_X>
# y_test = <votre_dataframe_test_y>

# Prédiction avec le modèle chargé
# y_pred_loaded = loaded_model.predict(X_test)

# Calcul des métriques pour le modèle chargé
# mse_loaded = mean_squared_error(y_test, y_pred_loaded)
# mae_loaded = mean_absolute_error(y_test, y_pred_loaded)
# r2_loaded = r2_score(y_test, y_pred_loaded)

# Affichage des métriques
# print("Performance du modèle chargé:")
# print(f"Mean Squared Error: {mse_loaded}")
# print(f"Mean Absolute Error: {mae_loaded}")
# print(f"R-squared: {r2_loaded}")

# Assurez-vous que les métriques recalculées sont proches de celles enregistrées dans le fichier de logs
