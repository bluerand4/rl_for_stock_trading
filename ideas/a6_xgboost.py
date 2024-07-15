#%%
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample DataFrame
data = {
    'Date': pd.date_range(start='1/1/2020', periods=10, freq='D'),
    'Open': np.random.randint(95, 105, size=10),
    'High': np.random.randint(105, 110, size=10),
    'Low': np.random.randint(90, 95, size=10),
    'Close': np.random.randint(95, 105, size=10),
    'Volume': np.random.randint(1000, 5000, size=10)
}
df = pd.DataFrame(data)

# Add a target variable 'Will_Close_Up'
df['Will_Close_Up'] = 0  # Initialize the column with 0
df['Will_Close_Up'][:-1] = (df['Close'].shift(-1) > df['Close']).astype(int)[:-1]  # Shift 'Close' and compare

# Define features and target
X = df[['Open', 'High', 'Low', 'Close', 'Volume']]  # Input features
y = df['Will_Close_Up']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display predictions and actual values
print("Predictions:", y_pred)
print("Actual:", y_test.tolist())

# %%
