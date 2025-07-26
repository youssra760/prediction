import pandas as pd
import numpy as np
import os
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 🔐 Secrets d'authentification
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REFRESH_TOKEN = os.getenv("REFRESH_TOKEN")

# 📊 URL publique du Google Sheet
csv_url = "https://docs.google.com/spreadsheets/d/18HmHLnT3fQrrV22zs_0bAym_VQZfG8zg/export?format=csv&gid=879814994"

def get_access_token():
    """Échange le refresh token contre un access token"""
    token_url = "https://oauth2.googleapis.com/token"
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "refresh_token": REFRESH_TOKEN,
        "grant_type": "refresh_token"
    }
    response = requests.post(token_url, data=payload)
    response.raise_for_status()
    return response.json()['access_token']

def upload_to_drive(filename, access_token):
    """Uploader le fichier Excel dans le Drive racine"""
    headers = {
        "Authorization": f"Bearer {access_token}"
    }

    metadata = {
        "name": filename,
        "mimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    }

    files = {
        'data': ('metadata', str(metadata), 'application/json'),
        'file': (filename, open(filename, "rb"), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    }

    upload_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart"
    response = requests.post(upload_url, headers=headers, files=files)

    if response.status_code in [200, 201]:
        print("✅ Fichier uploadé dans Google Drive.")
    else:
        print("❌ Échec de l'upload :", response.text)

def main():
    # Étape 1 : Charger les données depuis le Google Sheets
    df = pd.read_csv(csv_url)
    print(f"✅ Données chargées. Nombre de lignes : {len(df)}")

    # Étape 2 : Vérification des colonnes
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"❌ Colonne manquante : {col}")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    # Étape 3 : Modèle
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']
    model = LinearRegression()
    model.fit(X, y)
    df['Close_Predicted'] = model.predict(X)

    # Étape 4 : Prédiction du lendemain
    last_row = df.iloc[-1]
    next_day_features = np.array([[last_row['Open'], last_row['High'], last_row['Low'], last_row['Volume']]])
    next_day_pred = model.predict(next_day_features)[0]
    print(f"📈 Prédiction du prix de clôture du lendemain : {next_day_pred:.2f}")

    # Étape 5 : Métriques
    df['Absolute_Error'] = np.abs(df['Close'] - df['Close_Predicted'])
    df['Squared_Error'] = (df['Close'] - df['Close_Predicted']) ** 2
    df['Percentage_Error'] = df['Absolute_Error'] / df['Close'] * 100

    mae = mean_absolute_error(y, df['Close_Predicted'])
    rmse = np.sqrt(mean_squared_error(y, df['Close_Predicted']))
    r2 = r2_score(y, df['Close_Predicted'])

    print(f"\n🔢 Métriques globales :")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R² : {r2:.4f}")

    # Étape 6 : Enregistrer localement
    filename = "result.xlsx"
    df.to_excel(filename, index=False)
    print("📁 Fichier enregistré localement.")

    # Étape 7 : Upload Google Drive
    access_token = get_access_token()
    upload_to_drive(filename, access_token)

if __name__ == "__main__":
    main()
