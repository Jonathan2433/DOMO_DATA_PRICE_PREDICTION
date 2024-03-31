from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


class BaselineFiter(BaseEstimator, TransformerMixin):
    """
    Classe pour encapsuler le prétraitement et l'entraînement d'un modèle de régression forestière.

    Paramètres:
    ----------
    encoder : OneHotEncoder
        L'encodeur pour les variables catégorielles.
    scaler : RobustScaler
        Le scaler pour les variables numériques.
    model : RandomForestRegressor
        Le modèle de régression forestière à entraîner.
    """


    def __init__(self):
        self.encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.scaler = RobustScaler()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

    def fit(self, X):
        """
        Applique le prétraitement et entraîne le modèle sur les données fournies.

        Paramètres:
        ----------
        X : array-like
            Les données d'entraînement.
        y : array-like
            Les étiquettes cibles.

        Retourne:
        --------
        self : object
        """
        # Prétraitement
        # Supposons que 'y' soit la première colonne de X_transformed
        y = X[:, 1]
        X = np.delete(X, 1, axis=1)  # Supprimer la colonne 'y'

        # Appliquer l'encodage OneHot
        X_encoded = self._hot_encode(X)

        # Séparation des données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.33, random_state=42)

        # Scaling sur l'ensemble d'entraînement
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)

        self.model.fit(X_train_scaled, y_train)

        # Stocker X_test et y_test pour l'évaluation
        self.X_test = X_test  # Noter que X_test doit aussi être transformé avant l'évaluation
        self.y_test = y_test

        return self

    def predict(self, X):
        """
        Fait des prédictions avec le modèle entraîné sur les données prétraitées.

        Paramètres:
        ----------
        X : array-like
            Les données sur lesquelles faire des prédictions.

        Retourne:
        --------
        y_pred : array
            Les prédictions du modèle.
        """
        # X_encoded = self._hot_encode(X, fit_encoder=False)
        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return y_pred

    def evaluate(self):
        """
        Évalue les performances du modèle sur l'ensemble de test.

        Retourne:
        --------
        metrics : dict
            Un dictionnaire contenant le MSE, MAE et R^2.
        """
        y_pred = self.predict(self.X_test)
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        metrics = {
            "Mean Squared Error": mse,
            "Mean Absolute Error": mae,
            "R-squared": r2
        }
        return metrics

    def _hot_encode(self, X, fit_encoder=True):
        """
        Applique le prétraitement sur les données X.

        Paramètres:
        ----------
        X : array-like
            Les données à prétraiter.
        fit_encoder : bool
            Si True, ajuste l'encodeur sur les données.

        Retourne:
        --------
        X_processed : array-like
            Les données prétraitées.
        """
        # Supposons que la colonne CP est à l'indice 2
        cp_column = X[:, 1].reshape(-1, 1)

        if fit_encoder:
            cp_encoded = self.encoder.fit_transform(cp_column)
        else:
            cp_encoded = self.encoder.transform(cp_column)

        # Suppression de la colonne CP originale et 'ValeurFonciere' de X
        # Assurez-vous de supprimer 'ValeurFonciere' en premier pour éviter le décalage d'indice.
        X = np.delete(X, 2, axis=1)  # Supprime la colonne CP après la suppression de 'ValeurFonciere'
        # X = np.delete(X, 1, axis=1)  # Supprime 'ValeurFonciere'

        # Concaténation avec les colonnes encodées
        X_processed = np.hstack((X, cp_encoded))

        return X_processed
