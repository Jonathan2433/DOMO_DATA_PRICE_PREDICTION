from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

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
with open('./MODELS/RF_RANDOM_SEARCH_20240516_102902.pkl', 'rb') as f:
    model = pickle.load(f)


# Endpoint pour faire des prédictions
@app.post("/predict/")
async def predict(request: PredictionRequest):
    # Transformation des données en format attendu par le modèle
    input_data = np.array([[request.annee, request.surface_terrain, request.surface_bati,
                            request.surface_carrez, request.nombre_de_piece, request.code_postal,
                            request.type_de_bien]])
    # Prédiction
    prediction = model.predict(input_data)

    # Retourner le résultat
    return {"predicted_price": prediction[0]}