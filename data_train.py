import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.metrics import r2_score, mean_absolute_error

# -------- LOAD DATA --------
df = pd.read_csv("teen_phone_addiction_dataset.csv")

# -------- REMOVE TEXT COLUMNS (IMPORTANT) --------
drop_cols = ["Name", "Student_ID", "User_Name"]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# -------- ONE-HOT ENCODING --------
df = pd.get_dummies(df, columns=[
    "Phone_Usage_Purpose",
    "Parental_Control"
])

# -------- FEATURES --------
features = [
    'Daily_Usage_Hours',
    'Phone_Checks_Per_Day',
    'Time_on_Social_Media',
    'Time_on_Gaming',
    'Sleep_Hours',
    'Exercise_Hours',
    'Anxiety_Level',
    'Academic_Performance',
    'Social_Interactions'
]

# Add encoded categorical columns
features += [col for col in df.columns if col.startswith("Phone_Usage_Purpose_")]
features += [col for col in df.columns if col.startswith("Parental_Control_")]

# -------- INPUT / OUTPUT --------
X = df[features]
y = df["Addiction_Level"]

# -------- DEBUG CHECK --------
print("Feature Data Types:")
print(X.dtypes)

# -------- 70:20:10 SPLIT --------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=2/3, random_state=42
)

# -------- TRAIN MODEL --------
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------- FEATURE IMPORTANCE --------
importance_df = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values(by="importance", ascending=False)

print("\nFeature Importance:\n")
print(importance_df)

# -------- SAVE FILES --------
joblib.dump(model, "model.pkl")
joblib.dump(features, "features.pkl")
joblib.dump(importance_df, "importance.pkl")

print("\nModel trained and saved successfully.")


from sklearn.metrics import r2_score, mean_absolute_error

# Predictions on test set
y_pred = model.predict(X_test)

# Print metrics
print("\nModel Performance:")
print("R2 Score:", round(r2_score(y_test, y_pred), 3))
print("MAE:", round(mean_absolute_error(y_test, y_pred), 3))
