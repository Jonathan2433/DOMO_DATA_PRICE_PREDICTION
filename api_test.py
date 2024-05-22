import requests

url = 'http://127.0.0.1:8000/predict/'
data = {
    "annee": 2024,
    "surface_terrain": 0,
    "surface_bati": 45,
    "surface_carrez": 42,
    "nombre_de_piece": 2,
    "code_postal": 33130,
    "type_de_bien": "Appartement"
}

# Envoi de la requête POST à l'API
response = requests.post(url, json=data)

# Affichage de la réponse
print(response.json())
