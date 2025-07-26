import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def preprocess_data(df):
    """Convertir les colonnes en float et créer les variables décalées (lag)."""
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    for col in ['open', 'high', 'low', 'volume']:
        df[f'{col}_lag1'] = df[col].shift(1)

    return df.dropna()

def train_and_predict(df_symbol):
    """Entraîner le modèle et générer les prédictions + performance."""
    df_symbol = preprocess_data(df_symbol)

    X = df_symbol[['open_lag1', 'high_lag1', 'low_lag1', 'volume_lag1']]
    y = df_symbol['close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Évaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"\n📈 Résultats pour le symbole : {df_symbol['symbol'].iloc[0]}")
    print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.2f}")
    print("Comparaison réelle/prédit :")
    for true_val, pred_val in zip(y_test[:5], y_pred[:5]):
        print(f"  Réel : {true_val:.2f} | Prédit : {pred_val:.2f}")

    # Prédiction pour le jour suivant
    last_row = df_symbol.iloc[-1]
    next_day_input = pd.DataFrame([{
        'open_lag1': last_row['open'],
        'high_lag1': last_row['high'],
        'low_lag1': last_row['low'],
        'volume_lag1': last_row['volume']
    }])
    next_day_close = model.predict(next_day_input)[0]
    print(f"\n🔮 Prédiction du prix de clôture pour le jour suivant : {next_day_close:.2f}")

    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Symbol': df_symbol['symbol'].iloc[0],
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    })

    return results_df

def upload_to_drive(filename, creds):
    """Upload ou mise à jour du fichier Excel sur Google Drive (racine)."""
    service = build("drive", "v3", credentials=creds)

    query = f"name='{filename}' and trashed=false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])

    media = MediaFileUpload(filename, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if files:
        file_id = files[0]['id']
        service.files().update(fileId=file_id, media_body=media).execute()
        print(f"✅ Fichier mis à jour sur Google Drive (ID: {file_id})")
    else:
        metadata = {'name': filename}
        file = service.files().create(body=metadata, media_body=media).execute()
        print(f"✅ Nouveau fichier créé sur Google Drive (ID: {file['id']})")

def main():
    # 🔗 Télécharger les données depuis Google Sheets en CSV
    sheet_url = "https://docs.google.com/spreadsheets/d/18HmHLnT3fQrrV22zs_0bAym_VQZfG8zg/export?format=csv&gid=1356640539"
    df = pd.read_csv(sheet_url)

    all_results = pd.DataFrame()
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol].copy()
        results = train_and_predict(symbol_df)
        all_results = pd.concat([all_results, results], ignore_index=True)

    # 💾 Sauvegarde des résultats
    output_file = "predictions_results.xlsx"
    all_results.to_excel(output_file, index=False)
    print(f"\n📂 Résultats enregistrés dans le fichier {output_file}")

    # 🔐 Authentification Google Drive
    creds = Credentials(
        None,
        refresh_token=os.getenv("REFRESH_TOKEN"),
        client_id=os.getenv("CLIENT_ID"),
        client_secret=os.getenv("CLIENT_SECRET"),
        token_uri="https://oauth2.googleapis.com/token"
    )
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())

    upload_to_drive(output_file, creds)

if __name__ == "__main__":
    main()
