import os
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# 🔁 Rafraîchir access token
def refresh_access_token(client_id, client_secret, refresh_token):
    token_url = 'https://oauth2.googleapis.com/token'
    payload = {
        'client_id': client_id,
        'client_secret': client_secret,
        'refresh_token': refresh_token,
        'grant_type': 'refresh_token'
    }

    r = requests.post(token_url, data=payload)
    r.raise_for_status()
    return r.json().get('access_token')

# 📤 Upload vers Google Drive
def upload_to_drive(filepath, filename, access_token):
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    params = {
        'name': filename,
        'mimeType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    }

    files = {
        'data': ('metadata', json.dumps(params), 'application/json'),
        'file': open(filepath, "rb")
    }

    response = requests.post(
        "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
        headers=headers,
        files=files
    )

    if response.status_code == 200:
        print(f"✅ Fichier {filename} uploadé avec succès.")
    else:
        print("❌ Erreur upload :", response.text)

# 📈 Modèle et prédiction
def train_model_for_symbol(df_symbol):
    cols = ['open', 'high', 'low', 'close', 'volume']
    for col in cols:
        df_symbol[col] = df_symbol[col].astype(str).str.replace(',', '.').astype(float)

    for col in ['open', 'high', 'low', 'volume']:
        df_symbol[f'{col}_lag1'] = df_symbol[col].shift(1)

    df_symbol = df_symbol.dropna()

    X = df_symbol[['open_lag1', 'high_lag1', 'low_lag1', 'volume_lag1']]
    y = df_symbol['close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Calcul des métriques
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Erreurs ligne à ligne
    errors_abs = np.abs(y_test - y_pred)
    errors_sq = (y_test - y_pred) ** 2

    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Symbol': df_symbol['symbol'].iloc[0],
        'Erreur_Absolue': errors_abs,
        'Erreur_Carree': errors_sq
    })

    # Ajouter les métriques globales à chaque ligne
    results['MAE'] = mae
    results['MSE'] = mse
    results['RMSE'] = rmse
    results['R2'] = r2

    # Prédiction du lendemain
    last_row = df_symbol.iloc[-1]
    next_day_features = pd.DataFrame({
        'open_lag1': [last_row['open']],
        'high_lag1': [last_row['high']],
        'low_lag1': [last_row['low']],
        'volume_lag1': [last_row['volume']],
    })
    next_day_pred = model.predict(next_day_features)[0]

    summary_row = pd.DataFrame([{
        'Actual': None,
        'Predicted': next_day_pred,
        'Symbol': df_symbol['symbol'].iloc[0],
        'Erreur_Absolue': None,
        'Erreur_Carree': None,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }])

    full_results = pd.concat([results, summary_row], ignore_index=True)
    return full_results

# 🎯 Main
def main():
    # 🔗 Source Google Sheet
    url = "https://docs.google.com/spreadsheets/d/18HmHLnT3fQrrV22zs_0bAym_VQZfG8zg/export?format=csv&gid=1356640539"
    df = pd.read_csv(url)

    final_df = pd.DataFrame()

    for symbol in df['symbol'].unique():
        df_symbol = df[df['symbol'] == symbol].copy()
        result = train_model_for_symbol(df_symbol)
        final_df = pd.concat([final_df, result], ignore_index=True)

    # 💾 Export du fichier Excel localement
    file_path = "bourse_prediction.xlsx"
    final_df.to_excel(file_path, index=False)

    # 🔐 Authentification et upload
    client_id = os.environ["CLIENT_ID"]
    client_secret = os.environ["CLIENT_SECRET"]
    refresh_token = os.environ["REFRESH_TOKEN"]
    access_token = refresh_access_token(client_id, client_secret, refresh_token)

    upload_to_drive(file_path, "bourse_prediction.xlsx", access_token)

if __name__ == "__main__":
    main()
