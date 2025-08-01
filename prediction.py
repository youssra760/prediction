import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload


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

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"=== Résultats pour le symbole : {df_symbol['symbol'].iloc[0]} ===")
    print(f"MAE : {mae:.2f}")
    print(f"MSE : {mse:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R² : {r2:.2f}")

    print("\nComparaison des valeurs réelles / prédites :")
    for true_val, pred_val in zip(y_test[:5], y_pred[:5]):
        print(f"Vrai: {true_val:.2f} — Prédit: {pred_val:.2f}")

    last_row = df_symbol.iloc[-1]
    next_day_features = pd.DataFrame({
        'open_lag1': [last_row['open']],
        'high_lag1': [last_row['high']],
        'low_lag1': [last_row['low']],
        'volume_lag1': [last_row['volume']],
    })

    next_day_pred = model.predict(next_day_features)
    print(f"\nPrédiction close pour le jour suivant : {next_day_pred[0]:.2f}\n")

    # Résultats avec date (corrigé avec loc)
    results = pd.DataFrame({
        'Date': df_symbol.loc[y_test.index, 'date'].values,
        'Actual': y_test.values,
        'Predicted': y_pred,
        'Symbol': df_symbol['symbol'].iloc[0],
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    })

    # Ligne de prédiction du jour suivant
    next_day = pd.to_datetime(df_symbol['date'].max()) + pd.Timedelta(days=1)
    next_day_row = pd.DataFrame({
        'Date': [str(next_day.date())],
        'Actual': [np.nan],
        'Predicted': [next_day_pred[0]],
        'Symbol': [df_symbol['symbol'].iloc[0]],
        'MAE': [mae],
        'MSE': [mse],
        'RMSE': [rmse],
        'R2': [r2]
    })

    results = pd.concat([results, next_day_row], ignore_index=True)

    return results


def upload_to_drive(filename, creds):
    service = build("drive", "v3", credentials=creds)

    # Recherche fichier existant à la racine
    query = f"name='{filename}' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get('files', [])

    media = MediaFileUpload(filename, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if items:
        file_id = items[0]['id']
        service.files().update(fileId=file_id, media_body=media).execute()
        print(f"Fichier mis à jour sur Google Drive (ID: {file_id})")
    else:
        file_metadata = {
            "name": filename,
            "mimeType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        }
        uploaded_file = service.files().create(body=file_metadata, media_body=media).execute()
        print(f"Nouveau fichier créé sur Google Drive (ID: {uploaded_file.get('id')})")


def main():
    # URL de l'export CSV Google Sheets
    url = "https://docs.google.com/spreadsheets/d/1zet2MRDEotTpDpW5zPCaXHoljBTSKUYDnx7ICNXsHEI/export?format=csv&gid=990018493"
    df = pd.read_csv(url)

    symbols = df['symbol'].unique()
    all_results = pd.DataFrame()

    for symbol in symbols:
        df_symbol = df[df['symbol'] == symbol].copy()
        results = train_model_for_symbol(df_symbol)
        all_results = pd.concat([all_results, results], ignore_index=True)

    filename = "predictions_results.xlsx"
    all_results.to_excel(filename, index=False)
    print(f"Résultats sauvegardés dans {filename}")

    # Authentification OAuth 2.0
    CLIENT_ID = os.getenv('CLIENT_ID')
    CLIENT_SECRET = os.getenv('CLIENT_SECRET')
    REFRESH_TOKEN = os.getenv('REFRESH_TOKEN')

    creds = Credentials(
        None,
        refresh_token=REFRESH_TOKEN,
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        token_uri="https://oauth2.googleapis.com/token"
    )

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    upload_to_drive(filename, creds)


if __name__ == "__main__":
    main()
