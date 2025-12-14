import pandas as pd
import psycopg2

try:
    conn = psycopg2.connect(
        host="localhost",
        port="5433",
        database="smartmeterr",
        user="postgres",
        password="12345678"
    )

    query = 'SELECT * FROM "smartmeterr";'
    df = pd.read_sql_query(query, conn)

    # =========================
    # ANALISIS DATA (TARUH DI SINI)
    # =========================
    print("\n===== SHAPE DATA =====")
    print(df.shape)

    print("\n===== NAMA KOLOM =====")
    print(df.columns)

    print("\n===== INFO DATAFRAME =====")
    print(df.info())

    print("\n===== MISSING VALUES =====")
    print(df.isnull().sum())

    print("\n===== STATISTIK DESKRIPTIF =====")
    print(df.describe())

    # ======================================
    # TAMBAHKAN INI UNTUK MENAMPILKAN SEMUA DATA
    # ======================================
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print("\n===== DATA =====")
    print(df)   # ganti df.head() → df

    conn.close()

except Exception as e:
    print("Terjadi error:", e)
