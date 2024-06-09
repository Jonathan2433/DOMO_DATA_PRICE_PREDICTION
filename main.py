from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

app = FastAPI()


# Définition du modèle de données entrantes
class PredictionRequest(BaseModel):
    annee: int
    surface_terrain: float
    surface_bati: float
    surface_carrez: float
    nombre_de_piece: int
    code_postal: int
    type_de_bien: str


# Chargement du modèle RandomForest
with open('./MODELS/RF_RANDOM_SEARCH_20240609_201517.pkl', 'rb') as f:
    model = pickle.load(f)


# Endpoint pour faire des prédictions
@app.post("/predict/")
async def predict(request: PredictionRequest):
    # Transformation des données en DataFrame
    input_data = pd.DataFrame([request.dict()])
    input_data.columns = ['DT_Annee', 'SurfaceTerrain', 'SurfaceBati', 'SurfaceCarrez', 'NombrePiecesPrincipales',
                          'CodePostal', 'TypeLocalName']

    # Prédiction
    prediction = model.predict(input_data)

    # Retourner le résultat
    return {"predicted_price": prediction[0]}