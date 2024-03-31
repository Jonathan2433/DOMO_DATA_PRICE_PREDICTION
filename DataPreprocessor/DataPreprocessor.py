import numpy as np
import csv
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats

class DataPreprocessor(BaseEstimator, TransformerMixin):
    """
    Cette classe prépare un dataset pour l'entraînement de modèles de machine learning.
    Elle applique diverses transformations pour nettoyer et transformer les données, y compris la gestion des outliers.
    """

    def __init__(self):
        # Initialisation de la classe sans paramètres spécifiques
        pass

    def fit(self, X, y=None):
        return self

    def read_csv_to_numpy(self, filename, delimiter=';', skip_header=True):
        with open(filename, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file, delimiter=delimiter)
            if skip_header:
                next(csv_reader)  # Skip the header row
            data = list(csv_reader)
        return np.array(data, dtype=object)  # dtype=object pour gérer différents types

    def transform(self, X):
        # Si X est un chemin de fichier, le lire comme un tableau NumPy
        if isinstance(X, str):
            X = self.read_csv_to_numpy(X, skip_header=True)

        # Sinon, supposer que X est déjà un tableau NumPy
        X_transformed = np.copy(X)

        # Application des transformations spécifiques
        X_transformed = self._convert_columns(X_transformed)
        X_transformed = self._fill_na(X_transformed)
        X_transformed = self.feature_engineering(X_transformed)
        X_transformed = self._handle_outliers(X_transformed)
        X_transformed = self._prepare_final_dataset(X_transformed)

        return X_transformed

    def _convert_columns(self, X):
        date_col_index = 1  # Indice de la colonne de dates
        cols_to_convert = [2, 3, 16, 18, 20, 22, 24, 26, 27, 28, 31]  # Exemple d'indices de colonnes à convertir
        for col_idx in cols_to_convert:
            # Remplacer 'NULL' par np.nan
            column = np.where(X[:, col_idx] == 'NULL', np.nan, X[:, col_idx])
            # Remplacer les virgules par des points et essayer de convertir en float
            column = np.char.replace(column.astype(str), ',', '.').astype(float)
            # Si vous voulez remplacer np.nan par 0 (ou une autre valeur)
            column = np.nan_to_num(column, nan=0.0)
            X[:, col_idx] = column

        # Gestion de la colonne de dates
        converted_dates = [
            datetime.strptime(date_str, '%Y-%m-%d').year if date_str not in ['NULL', '', 'nan', 'NaN'] else 0
            for date_str in X[:, date_col_index]
        ]

        # Remplacer directement dans la colonne (ou ajouter) les années extraites
        X[:, date_col_index] = converted_dates

        return X

    def _fill_na(self, X):
        fill_zero_col_indices = [2, 3, 31, 16, 18, 20, 22, 24, 26, 27, 28]  # Ajustez selon vos colonnes

        # Convertir les colonnes en flottant sauf la colonne de dates
        for col_idx in fill_zero_col_indices:
            # Remplacer 'NULL' par np.nan
            X[:, col_idx] = np.where(np.logical_or(X[:, col_idx] == 'NULL', X[:, col_idx] == 'NaN'), np.nan, X[:, col_idx])
            # Remplacer les virgules par des points et essayer de convertir en float
            X[:, col_idx] = np.char.replace(X[:, col_idx].astype(str), ',', '.').astype(float)
            # Remplacer np.nan par 0
            X[:, col_idx] = np.nan_to_num(X[:, col_idx], nan=0.0)  # Utiliser np.nan_to_num pour remplacer np.nan par 0
        return X

    def feature_engineering(self, X):
        # Création de la colonne 'SurfaceCarrez'
        surface_carrez_columns = X[:, [16, 18, 20, 22, 24]]  # Indices des colonnes SurfaceCarrez1erLot, SurfaceCarrez2emeLot, etc.
        surface_carrez_columns[surface_carrez_columns == 'NULL'] = 0  # Remplace 'NULL' par 0
        surface_carrez_columns = surface_carrez_columns.astype(float)  # Convertit en flottant
        surface_carrez = np.sum(surface_carrez_columns, axis=1)  # Somme des surfaces par lot
        X = np.column_stack((X, surface_carrez))  # Ajout de la nouvelle colonne 'SurfaceCarrez'

        # Suppression des colonnes SurfaceCarrez1erLot, SurfaceCarrez2emeLot, etc.
        X = np.delete(X, [16, 18, 20, 22, 24], axis=1)

        return X

    def _handle_outliers(self, X):
        # Indices ajustés pour les colonnes après toutes les transformations précédentes
        valeur_fonciere_index = 2
        surface_bati_index = 22
        surface_carrez_index = 27  # Supposé être la dernière colonne après feature_engineering
        nombre_pieces_index = 23

        # Initialisation de la liste pour collecter tous les indices des outliers
        all_outliers_indices = []

        # Traitement pour la surface bâtie
        all_outliers_indices.extend(self._calculate_outliers_indices(X, surface_bati_index, valeur_fonciere_index))

        # Traitement pour la surface Carrez
        all_outliers_indices.extend(self._calculate_outliers_indices(X, surface_carrez_index, valeur_fonciere_index))

        # Traitement pour le nombre de pièces
        all_outliers_indices.extend(self._calculate_outliers_indices_for_categorical(X, nombre_pieces_index))

        # Suppression des duplicatas dans la liste des indices des outliers
        all_outliers_indices = list(set(all_outliers_indices))

        # Filtrage des outliers
        X_filtered = np.delete(X, all_outliers_indices, axis=0)

        return X_filtered

    def _calculate_outliers_indices(self, X, feature_col_index, price_col_index):
        # Calcul du prix par unité
        feature_values = X[:, feature_col_index].astype(float)
        price_values = X[:, price_col_index].astype(float)
        price_per_unit = np.divide(price_values, feature_values, out=np.zeros_like(price_values),
                                   where=feature_values != 0)

        # Calcul des quartiles et de l'IQR
        q1 = np.percentile(price_per_unit, 25)
        q3 = np.percentile(price_per_unit, 75)
        iqr = q3 - q1

        # Détermination des bornes pour les outliers
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        # Identification des indices des outliers
        outliers_indices = np.where((price_per_unit < lower_bound) | (price_per_unit > upper_bound))

        return outliers_indices[0]

    def _calculate_outliers_indices_for_categorical(self, X, cat_col_index):
        # Cette fonction est conçue pour traiter les variables catégorielles comme le nombre de pièces
        # Elle utilise une approche basée sur les quartiles de la distribution des valeurs catégorielles
        values = X[:, cat_col_index].astype(float)

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outliers_indices = np.where((values < lower_bound) | (values > upper_bound))

        return outliers_indices[0]

    def _prepare_final_dataset(self, X):
        # Prépare le dataset final en excluant certaines colonnes et en gérant les valeurs manquantes
        col_indices_to_keep = [1, 2, 8, 21, 22, 23, 26, 27]  # Indices des colonnes à conserver
        df = X[:, col_indices_to_keep]

        # Convertir les colonnes nécessaires en float
        surface_bati_index = 3  # Indice de la colonne SurfaceReelleBati
        valeur_fonciere_index = 1  # Indice de la colonne ValeurFonciere
        df[:, surface_bati_index] = df[:, surface_bati_index].astype(float)
        df[:, valeur_fonciere_index] = df[:, valeur_fonciere_index].astype(float)

        # Suppression des lignes avec des valeurs manquantes dans certaines colonnes
        mask = (df[:, surface_bati_index] != 0) & (df[:, valeur_fonciere_index] != 0)
        df = df[mask]

        return df



