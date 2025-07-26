import os
import pandas as pd
import joblib

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ✅ Chargement du fichier bourses.xlsx
excel_filename = "bourses.xlsx"
df = pd.read_excel(excel_filename)

# ✅ Chargement du modèle de régression (déjà entraîné et exporté avec joblib)
model = joblib.load("model_regression.pkl")  # Ce fichier doit être dans le même dossier

# ✅ Sélection des features pour la prédiction
features = df[["open", "high", "low", "volume"]]
predicted_close = model.predict(features)

# ✅ Ajout de la colonne prédite dans le DataFrame
df["predicted_close"] = predicted_close

# ✅ Sauvegarde dans un nouveau fichier Excel
predicted_filename = "bourses_predictions.xlsx"
df.to_excel(predicted_filename, index=False)
print("✅ Fichier prédictif sauvegardé localement dans bourses_predictions.xlsx")

# ✅ Upload vers Google Drive
CLIENT_ID = os.environ.get("CLIENT_ID")
CLIENT_SECRET = os.environ.get("CLIENT_SECRET")
REFRESH_TOKEN = os.environ.get("REFRESH_TOKEN")

creds = Credentials(
    None,
    refresh_token=REFRESH_TOKEN,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    token_uri="https://oauth2.googleapis.com/token"
)

if creds and creds.expired and creds.refresh_token:
    creds.refresh(Request())

service = build("drive", "v3", credentials=creds)

# ✅ Recherche d’un fichier déjà existant
query = f"name='{predicted_filename}' and mimeType='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' and trashed=false"
results = service.files().list(q=query, fields="files(id, name)").execute()
items = results.get('files', [])

media = MediaFileUpload(predicted_filename, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if items:
    file_id = items[0]['id']
    updated_file = service.files().update(
        fileId=file_id,
        media_body=media
    ).execute()
    print(f"✅ Fichier mis à jour sur Google Drive (ID: {file_id})")
else:
    file_metadata = {
        "name": predicted_filename,
        "mimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    }
    uploaded_file = service.files().create(
        body=file_metadata,
        media_body=media
    ).execute()
    print(f"✅ Nouveau fichier créé sur Google Drive (ID: {uploaded_file.get('id')})")
