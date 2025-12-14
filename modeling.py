from sqlalchemy import create_engine
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import pickle

# ==== KONEKSI DATABASE ====
engine = create_engine(
    "postgresql+psycopg2://postgres:12345678@localhost:5433/smartmeterr"
)

# ==== BACA DATA ====
df = pd.read_sql_query('SELECT * FROM "smartmeterr";', engine)

print("===== DATA BERHASIL DIBACA =====")
print(df.columns)
print(df.dtypes)

# ==== PILIH TARGET ====
target = df["kWh"]   # <--- TARGETMU

# ==== PILIH FITUR (numerik saja) ====
df_num = df.select_dtypes(include=["int64", "float64"])

# Masukkan target kembali
df_num["target"] = target

# Pastikan kolom target ada
if "target" not in df_num.columns:
    raise Exception("Kolom target hilang!")

# ==== Pisahkan fitur & target ====
X = df.drop(columns=[
    "kWh",        # target
    "id",         # kolom yang bikin model hafal
    "date",       # datetime
    "valid_start",
    "valid_end",
    "transaction_start",
    "transaction_end"
])
y = df_num["target"]

# ==== Train Test Split ====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==== Scaling ====
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==== Model ====
models = {
    "Linear Regression": LinearRegression(),
    "Support Vector Regressor": SVR(kernel='rbf'),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42)
}

results = {}

# ==== TRAIN & EVALUASI ====
for name, model in models.items():
    print(f"\n===== Training {name} =====")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results[name] = {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}

    pickle.dump(model, open(f"{name.replace(' ', '_')}.pkl", "wb"))

# ==== HASIL ====
print("\n===== PERFORMANCE SCORE =====")
for name, score in results.items():
    print(f"\n{name}:")
    print(f"MSE  : {score['MSE']:.3f}")
    print(f"RMSE : {score['RMSE']:.3f}")
    print(f"MAE  : {score['MAE']:.3f}")
    print(f"R²   : {score['R²']:.3f}")

pickle.dump(scaler, open("scaler.pkl", "wb"))
print("\nModel & Scaler berhasil disimpan!")
