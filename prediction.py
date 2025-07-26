import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# --- Configuration Google API ---
CLIENT_ID = 'TON_CLIENT_ID'
CLIENT_SECRET = 'TON_CLIENT_SECRET'
REFRESH_TOKEN = 'TON_REFRESH_TOKEN'
SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_ID = 'TON_FOLDER_ID_OPTIONNEL'  # ou None si pas de dossier

# --- Fonction d'upload sur Google Drive ---
def upload_to_drive(filename, folder_id=None):
    creds = Credentials(
        None,
        refresh_token=REFRESH_TOKEN,
        token_uri='https://oauth2.googleapis.com/token',
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        scopes=SCOPES
    )

    service = build('drive', 'v3', credentials=creds)

    file_metadata = {'name': filename}
    if folder_id:
        file_metadata['parents'] = [folder_id]

    media = MediaFileUpload(filename, resumable=True)
    file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

    print(f"Fichier '{filename}' uploadé avec l'ID: {file.get('id')}")

# --- Fonction d'entrainement et prédiction par symbole ---
def train_model_for_symbol(df_symbol):
    cols = ['open', 'high', 'low', 'close', 'volume']
    for col in cols:
        df_symbol[col] = df_symbol[col].astype(str).str.replace(',', '.').astype(float)

    # Création des variables lag1
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

    # Retourner dataframe pour sauvegarde résultats
    results = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    })
    results['Symbol'] = df_symbol['symbol'].iloc[0]
    results['MAE'] = mae
    results['MSE'] = mse
    results['RMSE'] = rmse
    results['R2'] = r2

    return results

# --- Fonction principale ---
def main():
    url = "https://docs.google.com/spreadsheets/d/18HmHLnT3fQrrV22zs_0bAym_VQZfG8zg/export?format=csv&gid=1356640539"
    df = pd.read_csv(url)

    symbols = df['symbol'].unique()

    all_results = pd.DataFrame()

    for symbol in symbols:
        df_symbol = df[df['symbol'] == symbol].copy()
        results = train_model_for_symbol(df_symbol)
        all_results = pd.concat([all_results, results], ignore_index=True)

    # Sauvegarder résultats dans CSV
    filename = 'predictions_results.csv'
    all_results.to_csv(filename, index=False)
    print(f"Résultats sauvegardés dans {filename}")

    # Upload sur Google Drive
    upload_to_drive(filename, folder_id=FOLDER_ID)

if __name__ == "__main__":
    main()

