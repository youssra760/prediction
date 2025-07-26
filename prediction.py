import os, io, requests
import pandas as pd
from sklearn.linear_model import LinearRegression
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload

# 🔐 Charger les secrets depuis GitHub Actions
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")
SHEET_ID = os.getenv("EXCEL_FILE_ID")  # ID Google Sheet
FOLDER_ID = os.getenv("FOLDER_ID")     # ID dossier Drive cible

# 1. Rafraîchir le token OAuth
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

# 2. Vérification d'accès sécurisé (pour Shared Drives)
try:
    meta = service.files().get(
        fileId=SHEET_ID,
        fields="id,name,mimeType",
        supportsAllDrives=True
    ).execute()
    print("✅ Accès validé :", meta["name"], "(", meta["mimeType"], ")")
except Exception as e:
    print("❌ Accès impossible :", e)
    exit(1)

# 3. Export de la Google Sheet en fichier Excel
request = service.files().export_media(
    fileId=SHEET_ID,
    mimeType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
fh = io.BytesIO()
downloader = MediaIoBaseDownload(fh, request)
done = False
while not done:
    _, done = downloader.next_chunk()
fh.seek(0)

# 4. Traitement des données
df = pd.read_excel(fh).dropna()
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]
y = df['Close_tmr']

model = LinearRegression().fit(X, y)
pred_value = float(model.predict(X.iloc[-1].values.reshape(1, -1))[0])

# 5. Génération du fichier CSV avec prédiction
csv_filename = "prediction.csv"
pd.DataFrame([{
    'date': pd.Timestamp.today().date().isoformat(),
    'prediction_close': pred_value
}]).to_csv(csv_filename, index=False)
print("✔ Fichier CSV généré :", csv_filename)

# 6. Upload vers le dossier Drive cible
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
