import requests

url = 'http://127.0.0.1:8000/predict/'
data = {
    "annee": 2024,
    "surface_terrain": 0,
    "surface_bati": 77,
    "surface_carrez": 77,
    "nombre_de_piece": 4,
    "code_postal": 33300,
    "type_de_bien": "Appartement"
}

# Envoi de la requête POST à l'API
response = requests.post(url, json=data)

# Affichage de la réponse
print(response.json())
