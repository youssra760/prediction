import os, io, requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

# Lecture des secrets
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")
EXCEL_FILE_ID = os.getenv("EXCEL_FILE_ID")
FOLDER_ID = os.getenv("FOLDER_ID")  # <-- Utilisé pour uploader

# Authentification OAuth2
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

# Téléchargement du fichier source Excel ou Sheets
request = service.files().export_media(
    fileId=EXCEL_FILE_ID,
    mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
fh = io.BytesIO()
downloader = MediaIoBaseDownload(fh, request)
done = False
while not done:
    _, done = downloader.next_chunk()
fh.seek(0)

df = pd.read_excel(fh).dropna()
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
y = df['Close_tmr']
model = LinearRegression().fit(X, y)

pred_value = float(model.predict(X.iloc[-1].values.reshape(1, -1))[0])

# Création du CSV
csv_filename = "prediction.csv"
pd.DataFrame([{'date': pd.Timestamp.today().date().isoformat(),
               'prediction_close': pred_value}]).to_csv(csv_filename, index=False)
print(f"Fichier CSV généré : {csv_filename}")

# Upload sur Google Drive dans le FOLDER_ID
file_metadata = {
    'name': csv_filename,
    'parents': [FOLDER_ID],
    'mimeType': 'text/csv'
}
media = MediaIoBaseUpload(open(csv_filename, 'rb'), mimetype='text/csv')
uploaded = service.files().create(body=file_metadata, media_body=media, fields="id").execute()
print("Upload réussi, file ID:", uploaded.get("id"))
