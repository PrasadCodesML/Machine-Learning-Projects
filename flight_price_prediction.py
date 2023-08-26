import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

url = "https://raw.githubusercontent.com/azmathmoosa/Indian-Flight-Price-Prediction/master/Data_Train.csv"
df = pd.read_csv(url)

df['Journey_Day'] = pd.to_datetime(df['Date_of_Journey']).dt.day
df['Journey_Month'] = pd.to_datetime(df['Date_of_Journey']).dt.month

df['Dep_Hour'] = pd.to_datetime(df['Dep_Time']).dt.hour
df['Dep_Min'] = pd.to_datetime(df['Dep_Time']).dt.minute

df['Arrival_Hour'] = pd.to_datetime(df['Arrival_Time']).dt.hour
df['Arrival_Min'] = pd.to_datetime(df['Arrival_Time']).dt.minute

df['Duration_Hours'] = df['Duration'].str.extract('(\d+)h').fillna(0).astype(int)
df['Duration_Mins'] = df['Duration'].str.extract('(\d+)m').fillna(0).astype(int)
df['Total_Duration'] = df['Duration_Hours']*60 + df['Duration_Mins']

df = df.drop(['Date_of_Journey', 'Dep_Time', 'Arrival_Time', 'Duration', 
              'Duration_Hours', 'Duration_Mins', 'Route', 'Additional_Info'], axis=1)

categorical_features = ['Airline', 'Source', 'Destination']

le = LabelEncoder()
for feature in categorical_features:
    df[feature] = le.fit_transform(df[feature])

df['Total_Stops'] = df['Total_Stops'].fillna(df['Total_Stops'].mode()[0])
df = df.dropna()

X = df.drop(['Price'], axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

y_pred = rf_model.predict(X_test_scaled)

print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x='importance', y='feature', data=feature_importance)
plt.title('Feature Importance in Flight Price Prediction')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Prices (INR)')
plt.ylabel('Predicted Prices (INR)')
plt.title('Actual vs Predicted Flight Prices')
plt.tight_layout()
plt.show() 