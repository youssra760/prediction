import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import smtplib
from email.message import EmailMessage

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

    return results

def send_email_with_attachment(sender_email, sender_password, receiver_email, subject, body, filename):
    msg = EmailMessage()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject
    msg.set_content(body)

    with open(filename, 'rb') as f:
        file_data = f.read()
        file_name = f.name

    msg.add_attachment(file_data, maintype='application', subtype='vnd.openxmlformats-officedocument.spreadsheetml.sheet', filename=file_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)

    print(f"Email envoyé à {receiver_email} avec la pièce jointe {filename}")

def main():
    url = "https://docs.google.com/spreadsheets/d/18HmHLnT3fQrrV22zs_0bAym_VQZfG8zg/export?format=csv&gid=1356640539"
    df = pd.read_csv(url)

    symbols = df['symbol'].unique()
    all_results = pd.DataFrame()

    for symbol in symbols:
        df_symbol = df[df['symbol'] == symbol].copy()
        results = train_model_for_symbol(df_symbol)
        all_results = pd.concat([all_results, results], ignore_index=True)

    filename = 'predictions_results.xlsx'
    all_results.to_excel(filename, index=False)
    print(f"Résultats sauvegardés dans {filename}")

    # Récupération des variables d'environnement
    sender_email = os.getenv('SENDER_EMAIL')
    sender_password = os.getenv('SENDER_PASSWORD')
    receiver_email = os.getenv('RECEIVER_EMAIL')

    print(f"SENDER_EMAIL : {sender_email}")
    print(f"SENDER_PASSWORD : {'[secret]' if sender_password else None}")
    print(f"RECEIVER_EMAIL : {receiver_email}")

    if sender_email and sender_password and receiver_email:
        subject = 'Résultats des prédictions'
        body = 'Bonjour,\n\nVeuillez trouver en pièce jointe le fichier Excel contenant les résultats des prédictions.\n\nCordialement.'
        send_email_with_attachment(sender_email, sender_password, receiver_email, subject, body, filename)
    else:
        print("Une ou plusieurs variables d'environnement pour l'email sont manquantes. Email non envoyé.")

if __name__ == "__main__":
    main()
