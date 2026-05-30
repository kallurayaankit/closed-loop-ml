import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)
n = 1000
data = {
    'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n)],
    'product_category': np.random.randint(0, 10, n),
    'price': np.random.uniform(10, 200, n),
    'day_of_week': np.random.randint(0, 7, n),
    'demand': np.random.poisson(lam=20, size=n) + np.random.uniform(0, 10, n)
}
df = pd.DataFrame(data)
df.to_csv('data/training_data.csv', index=False)
print("Training data created.")