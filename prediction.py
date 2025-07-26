import os
import io
import requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload
from googleapiclient.errors import HttpError

# Charger les variables d'environnement
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")
EXCEL_FILE_ID = os.getenv("EXCEL_FILE_ID")
FOLDER_ID = os.getenv("FOLDER_ID")

# Vérifier que les variables d'environnement sont définies
if not all([CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN, EXCEL_FILE_ID, FOLDER_ID]):
    raise ValueError("Une ou plusieurs variables d'environnement manquent (CLIENT_ID, CLIENT_SECRET, REFRESH_TOKEN, EXCEL_FILE_ID, FOLDER_ID)")

# Authentification
try:
    resp = requests.post(
        "https://oauth2.googleapis.com/token",
        data={
            "grant_type": "refresh_token",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "refresh_token": REFRESH_TOKEN,
        }
    )
    resp.raise_for_status()
    creds = Credentials(token=resp.json()["access_token"])
    service = build("drive", "v3", credentials=creds)
except Exception as e:
    print(f"Erreur lors de l'authentification : {e}")
    raise

# Vérifier si le fichier existe et est exportable
try:
    file_metadata = service.files().get(fileId=EXCEL_FILE_ID, fields="id, name, mimeType").execute()
    print(f"Fichier trouvé : {file_metadata['name']} (Type: {file_metadata['mimeType']})")
    
    # Vérifier si le fichier est un Google Sheets
    if file_metadata["mimeType"] != "application/vnd.google-apps.spreadsheet":
        raise ValueError(f"Le fichier (ID: {EXCEL_FILE_ID}) n'est pas un Google Sheets. Type trouvé : {file_metadata['mimeType']}")
except HttpError as e:
    print(f"Erreur lors de la vérification du fichier : {e}")
    if "404" in str(e):
        print(f"Le fichier avec l'ID {EXCEL_FILE_ID} n'existe pas ou n'est pas accessible.")
    raise

# Exporter le fichier
try:
    request = service.files().export_media(
        fileId=EXCEL_FILE_ID,
        mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Téléchargement : {int(status.progress() * 100)}%")
    fh.seek(0)
except HttpError as e:
    print(f"Erreur lors de l'exportation du fichier : {e}")
    raise

# Traiter les données et générer la prédiction
try:
    df = pd.read_excel(fh).dropna()
    X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    y = df['Close_tmr']
    model = LinearRegression().fit(X, y)
    pred_value = float(model.predict(X.iloc[-1].values.reshape(1, -1))[0])
except Exception as e:
    print(f"Erreur lors du traitement des données ou de la prédiction : {e}")
    raise

# Sauvegarder la prédiction dans un fichier CSV
csv_filename = "prediction.csv"
try:
    pd.DataFrame([{
        'date': pd.Timestamp.today().date().isoformat(),
        'prediction_close': pred_value
    }]).to_csv(csv_filename, index=False)
    print(f"✔ CSV généré : {csv_filename}")
except Exception as e:
    print(f"Erreur lors de la génération du CSV : {e}")
    raise

# Uploader le fichier CSV sur Google Drive
try:
    file_metadata = {
        'name': csv_filename,
        'parents': [FOLDER_ID],
        'mimeType': 'text/csv'
    }
    media = MediaIoBaseUpload(open(csv_filename, 'rb'), mimetype='text/csv')
    uploaded = service.files().create(
        body=file_metadata,
        media_body=media,
        fields="id",
        supportsAllDrives=True
    ).execute()
    print(f"✅ Upload réussi ID : {uploaded.get('id')}")
except HttpError as e:
    print(f"Erreur lors de l'upload du fichier : {e}")
    raise
