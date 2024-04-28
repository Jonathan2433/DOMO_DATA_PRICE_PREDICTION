
# Utilise une image de base officielle Python
FROM python:3.9-slim

# Définis le répertoire de travail dans le conteneur
WORKDIR /code

# Copie les fichiers de dépendances dans le répertoire de travail
COPY ./requirements.txt /code/requirements.txt

# Copie les fichiers de ton application dans le répertoire de travail du conteneur
COPY main.py ./
COPY app ./app

# Copie le fichier du modèle dans le répertoire de travail du conteneur
COPY ./MODELS/BASELINE/RF_REGRESSOR_BASELINE_20240426_141536.pkl /code/MODELS/BASELINE/


# Installe les dépendances
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copie les fichiers de ton projet dans le répertoire de travail
COPY ./app /code/app

# Informe Docker que l'application écoute sur le port 8000
EXPOSE 8000

# Exécute l'application FastAPI avec Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
