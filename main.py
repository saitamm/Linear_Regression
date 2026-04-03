import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# 1. Load data
df = pd.read_csv('Student_Performance_2026.csv')

# 2. Define X and y
X = df[['daily_screen_time_hours', 'online_study_hours', 'gaming_hours',
        'sleep_hours', 'attendance_percentage', 'offline_study_hours',
        'previous_sem_CGPA']]
y = df['current_sem_CGPA']

# 3. Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training students : {X_train.shape[0]}")
print(f"Testing students  : {X_test.shape[0]}")

# 4. Train
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate
print(f"\nR²   : {r2_score(y_test, y_pred):.4f}")
print(f"MAE  : {mean_absolute_error(y_test, y_pred):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

# 7. Coefficients
coef_df = pd.DataFrame({
    'Feature'    : X.columns,
    'Coefficient': model.coef_
}).sort_values('Coefficient', ascending=False)

print("\nCoefficients:")
print(coef_df.to_string(index=False))
print(f"\nIntercept: {model.intercept_:.4f}")

# 8. Save
joblib.dump(model, 'model_linear.pkl')
print("\n✅ model_linear.pkl saved!")