from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

def train_model_for_symbol(df_symbol):
    # ... ton preprocessing + training ici ...
    
    # Prédictions sur le test set
    y_pred = model.predict(X_test)
    
    # Calcul des métriques
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Prédiction du jour suivant
    next_day_pred = model.predict(X_latest)[0]

    # 1️⃣ Résultats des prédictions classiques
    results = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Symbol': df_symbol['symbol'].iloc[0],
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    })

    # 2️⃣ Ligne spéciale pour la prédiction du lendemain
    summary_row = pd.DataFrame([{
        'Actual': None,
        'Predicted': next_day_pred,
        'Symbol': df_symbol['symbol'].iloc[0],
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }])

    # 3️⃣ Fusionner les deux
    full_results = pd.concat([results, summary_row], ignore_index=True)

    return full_results
