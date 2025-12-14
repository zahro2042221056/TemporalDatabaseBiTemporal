import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import pickle
import numpy as np

# ===== CONFIG =====
st.set_page_config(page_title="SAFIMETRIC 📊💖", layout="wide")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F19CBB;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #f7bfd4;
    }

    /* Judul */
    h1, h2, h3, h4, h5, h6 {
        color: #4a0033;
    }

    /* Teks normal */
    .css-10trblm, .css-1cpxqw2 {
        color: black;
    }

    /* Tombol */
    button {
        background-color: #d63384 !important;
        color: white !important;
        border-radius: 8px;
        font-weight: bold;
    }

    /* Dataframe header */
    thead tr th {
        background-color: #e577a3 !important;
        color: white !important;
    }

    /* Input box */
    input, textarea {
        border-radius: 8px !important;
        border: 2px solid #d63384 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ===== KONEKSI DATABASE =====
engine = create_engine("postgresql+psycopg2://postgres:12345678@localhost:5433/smartmeterr")

# ===== LOAD DATA =====
df = pd.read_sql_query('SELECT * FROM "smartmeterr"', engine)
df['date'] = pd.to_datetime(df['date'])

# ===== LOAD MODEL =====
model_rf = pickle.load(open("Random_Forest.pkl", "rb"))
model_lr = pickle.load(open("Linear_Regression.pkl", "rb"))
model_svr = pickle.load(open("Support_Vector_Regressor.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ===== SIDEBAR =====
st.sidebar.title("Menu")
menu = st.sidebar.radio("Pilih Halaman:", ["Overview Data", "EDA", "ML Prediction", "Temporal Query"])

# ======================
# ===== PAGE 1 =====
# ======================
if menu == "Overview Data":
    st.title("📊 Bitemporal Smart Meter Database")
    st.write("Preview dataset dari PostgreSQL:")
    st.dataframe(df.head(10))
    st.write("Statistik Deskriptif:")
    st.write(df[['kWh', 'voltage']].describe())

# ======================
# ===== PAGE 2 =====
# ======================
elif menu == "EDA":
    st.title("📈 Exploratory Data Analysis")

    st.subheader("Tren kWh dari Waktu ke Waktu")
    fig1, ax1 = plt.subplots()
    ax1.plot(df['date'], df['kWh'])
    st.pyplot(fig1)

    st.subheader("Tren Voltage dari Waktu ke Waktu")
    fig2, ax2 = plt.subplots()
    ax2.plot(df['date'], df['voltage'])
    st.pyplot(fig2)

    st.subheader("Histogram Konsumsi kWh")
    fig3, ax3 = plt.subplots()
    ax3.hist(df['kWh'], bins=20)
    st.pyplot(fig3)

    st.subheader("Heatmap Korelasi Fitur")
    fig4, ax4 = plt.subplots()
    corr = df[['kWh','voltage']].corr()
    im = ax4.imshow(corr, cmap="coolwarm")
    ax4.set_xticks(range(len(corr)))
    ax4.set_yticks(range(len(corr)))
    ax4.set_xticklabels(corr.columns)
    ax4.set_yticklabels(corr.columns)
    plt.colorbar(im)
    st.pyplot(fig4)

# ======================
# ===== PAGE 3 =====
# ======================
elif menu == "ML Prediction":
    st.title("🤖 Machine Learning Prediction (kWh Estimation)")

    voltage = st.number_input("Masukkan nilai Voltage:", min_value=200.0, max_value=300.0, step=0.1)

    if st.button("Prediksi Konsumsi kWh"):
        data = np.array([[voltage]])
        data_scaled = scaler.transform(data)

        pred_lr = model_lr.predict(data_scaled)[0]
        pred_svr = model_svr.predict(data_scaled)[0]
        pred_rf = model_rf.predict(data_scaled)[0]

        st.success("✅ Hasil Prediksi:")
        st.write(f"Linear Regression : {pred_lr:.6f}")
        st.write(f"Support Vector Reg. : {pred_svr:.6f}")
        st.write(f"Random Forest : {pred_rf:.6f}")

# ======================
# ===== PAGE 4 =====
# ======================
elif menu == "Temporal Query":
    st.title("⏳ Temporal Query Interface")

    date_input = st.date_input("Pilih Tanggal (Application Time Query)")

    query = f"""
    SELECT *
    FROM "smartmeterr"
    WHERE '{date_input}' BETWEEN valid_start AND valid_end
    """

    result = pd.read_sql_query(query, engine)

    st.write("Hasil Query Bitemporal:")
    st.dataframe(result)
