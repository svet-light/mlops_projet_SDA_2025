# Utiliser une image de base Python
FROM python:3.9-slim

# Définir le répertoire de travail dans le conteneur
WORKDIR /app

# Copier les fichiers nécessaires dans le conteneur
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirement.txt

# Exposer le port sur lequel l'application Flask va écouter
EXPOSE 5000

# Commande pour lancer l'application Flask
CMD ["python", "app.py"]