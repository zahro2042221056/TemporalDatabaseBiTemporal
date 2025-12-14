import pandas as pd
from sqlalchemy import create_engine

# KONEKSI DATABASE
engine = create_engine("postgresql+psycopg2://postgres:12345678@localhost:5433/smartmeterr")

# BACA DATA
df = pd.read_sql_query('SELECT * FROM "smartmeterr";', engine)

# CONVERT & HILANGKAN TIMEZONE
datetime_cols = ['valid_start', 'valid_end', 'transaction_start', 'transaction_end']

for col in datetime_cols:
    df[col] = pd.to_datetime(df[col]).dt.tz_localize(None)
    
# PREPROCESSING
df['Date'] = pd.to_datetime(df['date'])
df['valid_start'] = pd.to_datetime(df['valid_start'])
df['valid_end'] = pd.to_datetime(df['valid_end'])
df['transaction_start'] = pd.to_datetime(df['transaction_start'])
df['transaction_end'] = pd.to_datetime(df['transaction_end'])

print("===== DATA SIAP DIPROSES =====")
print(df.head())

# ======================
# 2. BUAT LABEL BACKDATED (TARO DI SINI)
# ======================
df['is_backdated'] = (df['transaction_start'] > df['valid_start']).astype(int)

print("===== CEK LABEL BACKDATED =====")
print(df[['valid_start', 'transaction_start', 'is_backdated']].head())