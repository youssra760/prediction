import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# === Connexion à Google Sheets
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
gc = gspread.authorize(credentials)

# Remplacer par ton ID Google Sheets
sheet_url = "https://docs.google.com/spreadsheets/d/1zet2MRDEotTpDpW5zPCaXHoljBTSKUYDnx7ICNXsHEI/edit?usp=sharing"
spreadsheet = gc.open_by_url(sheet_url)
worksheet = spreadsheet.get_worksheet(0)  # 0 = première feuille

# Charger les données en DataFrame
data = worksheet.get_all_records()
df = pd.DataFrame(data)

# === Ajouter une colonne symbol si elle n’existe pas
if 'symbol' not in df.columns:
    df['symbol'] = 'Bourse'

# === Lancer le modèle
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

    results = pd.DataFrame({
        'Actual': y_test.values,
        'Predicted': y_pred
    })
    results['Symbol'] = df_symbol['symbol'].iloc[0]
    results['MAE'] = mae
    results['MSE'] = mse
    results['RMSE'] = rmse
    results['R2'] = r2

    next_day = pd.to_datetime(df_symbol['date'].max()) + pd.Timedelta(days=1)
    next_day_row = pd.DataFrame({
        'Actual': [np.nan],
        'Predicted': [next_day_pred[0]],
        'Symbol': [df_symbol['symbol'].iloc[0]],
        'MAE': [mae],
        'MSE': [mse],
        'RMSE': [rmse],
        'R2': [r2]
    }, index=[str(next_day.date())])

    results = pd.concat([results, next_day_row])

    return results

# === Exécution
train_model_for_symbol(df)
