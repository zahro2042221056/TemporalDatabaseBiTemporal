import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt

# Koneksikan lagi
engine = create_engine("postgresql+psycopg2://postgres:12345678@localhost:5433/smartmeterr")

# Ambil data
df = pd.read_sql_query('SELECT * FROM "smartmeterr";', engine)

# Convert date
df['date'] = pd.to_datetime(df['date'])

print("===== SHAPE =====")
print(df.shape)

print("\n===== Statistik smartmeterr =====")
print(df['date'].describe())
print(df['kWh'].describe())
print(df['voltage'].describe())

# ---------- GRAFIK 1: Tren smartmeterr dari waktu ke waktu ----------
plt.figure(figsize=(10,5))
plt.plot(df['date'], df['kWh'])
plt.title("Tren kWh dari Waktu ke Waktu")
plt.xlabel("Tanggal")
plt.ylabel("kWh")
plt.grid(True)
plt.show()

plt.figure(figsize=(10,5))
plt.plot(df['date'], df['voltage'])
plt.title("Tren Voltage dari Waktu ke Waktu")
plt.xlabel("Tanggal")
plt.ylabel("Volt")
plt.grid(True)
plt.show()

# ---------- GRAFIK 2: Histogram smartmeterr ----------
plt.figure(figsize=(7,5))
plt.hist(df['kWh'], bins=20)
plt.title("Distribusi smartmeterr")
plt.xlabel("smartmeterr")
plt.ylabel("Frekuensi")
plt.grid(True)
plt.show()

# ---------- GRAFIK 3: Korelasi numerik ----------
plt.figure(figsize=(10,7))
numeric_df = df.select_dtypes(include=['int64','float64'])
plt.matshow(numeric_df.corr())
plt.title("Heatmap Korelasi Fitur Numerik", pad=50)
plt.colorbar()
plt.show()
