import os
import numpy as np
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

def retrain_challenger():
    df = pd.read_csv('data/training_data.csv')
    # Simulate new data by adding noise
    df['demand'] = df['demand'] + pd.Series(np.random.normal(0, 2, len(df)))
    X = df[['product_category', 'price', 'day_of_week']]
    y = df['demand']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, 'models/challenger.pkl')
    mae = mean_absolute_error(y_test, model.predict(X_test))
    return mae

def promote_challenger():
    os.replace('models/challenger.pkl', 'models/champion.pkl')
    print("Challenger promoted to champion!")

def rollback():
    print("Challenger rejected. Keeping current champion.")
    if os.path.exists('models/challenger.pkl'):
        os.remove('models/challenger.pkl')

def alert(message):
    print(f"🚨 ALERT: {message}")
    # In production, send Slack/email here

# --- Main loop ---
with open('data/mae.txt', 'r') as f:
    current_mae = float(f.read())

THRESHOLD = 5.0  # example MAE threshold

print(f"Current champion MAE: {current_mae:.2f}")

if current_mae > THRESHOLD:
    alert("Performance degradation detected! Starting retraining...")
    challenger_mae = retrain_challenger()
    print(f"Challenger MAE: {challenger_mae:.2f}")

    if challenger_mae < current_mae:
        promote_challenger()
    else:
        rollback()
else:
    print("Model performance is healthy. No action needed.")