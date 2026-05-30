import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('data/training_data.csv')
X = df[['product_category', 'price', 'day_of_week']]
y = df['demand']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
joblib.dump(model, 'models/champion.pkl')
print(f"Champion model trained. MAE: {mean_absolute_error(y_test, model.predict(X_test)):.2f}")