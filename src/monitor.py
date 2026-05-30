import pandas as pd
import joblib

model = joblib.load('models/champion.pkl')
reference = pd.read_csv('data/training_data.csv')

# Simulate live data (slightly different distribution)
live_data = reference.sample(200).copy()
live_data['demand'] = live_data['demand'] * 1.2  # simulate shift

X_live = live_data[['product_category', 'price', 'day_of_week']]
y_live = live_data['demand']
preds = model.predict(X_live)

mae = abs(y_live - preds).mean()
print(f"Realized MAE: {mae:.2f}")

# Save for orchestrator
with open('data/mae.txt', 'w') as f:
    f.write(str(mae))