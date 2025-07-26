import os, io, requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# Lire les secrets depuis GitHub Actions
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")
EXCEL_FILE_ID = os.getenv("EXCEL_FILE_ID")  # ID de ta Google Sheet
FOLDER_ID = os.getenv("FOLDER_ID")          # ID du dossier cible Drive

# 1. Rafraîchissement du token
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

# 2. Vérifier l'accès à la feuille
try:
    meta = service.files().get(
        fileId=EXCEL_FILE_ID,
        fields="id,name,mimeType",
        supportsAllDrives=True
    ).execute()
    print("Accès confirmé :", meta.get("name"), "(", meta.get("mimeType"), ")")
except Exception as e:
    print("Erreur accès au fichier :", e)
    exit(1)

# 3. Export de la Google Sheet (en XLSX)
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

# 4. Lire les données
df = pd.read_excel(fh).dropna()
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
y = df['Close_tmr']

# 5. Modèle de régression linéaire
model = LinearRegression().fit(X, y)
pred = float(model.predict(X.iloc[-1].values.reshape(1, -1))[0])

# 6. Générer le fichier CSV
csv_filename = "prediction.csv"
pd.DataFrame([{'date': pd.Timestamp.today().date().isoformat(),
               'prediction_close': pred}]).to_csv(csv_filename, index=False)
print("✔ CSV généré :", csv_filename)

# 7. Upload vers Drive (même pour Shared Drive)
file_metadata = {
    'name': csv_filename,
    'parents': [FOLDER_ID],
    'mimeType': 'text/csv'
}
media = MediaFileUpload(csv_filename, mimetype='text/csv')
uploaded = service.files().create(
    body=file_metadata,
    media_body=media,
    supportsAllDrives=True,
    fields="id"
).execute()
print("✅ Upload réussi, ID :", uploaded.get("id"))
