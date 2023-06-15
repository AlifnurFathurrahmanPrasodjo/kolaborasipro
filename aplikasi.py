import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import calendar


# Mengimpor model dan text dari file yang telah diekspor sebelumnya
svr = joblib.load('modelSVM.pkl')
df = pd.read_csv('DatasetSaham.csv')
scaler = joblib.load('scalerUAS.pkl')
dataset_test = pd.read_csv('data-test-UAS.csv')




# Judul aplikasi
st.markdown("<h1 style='text-align: center;'>Forecasting Harga Saham PT. INDOFOOD</h1>", unsafe_allow_html=True)
# st.title('Forecasting Temperature Anomalies Data Time Series')


# Tombol untuk memprediksi
if st.button('PREDICT'):
    
    col1, col2 = st.columns(2)

    tanggal = df['Date'].tail(1).values
    tanggal = tanggal[0].split('-')
    tahun = tanggal[0]
    bulan = tanggal[1]
    hari = tanggal[2]
    tahun = int(tahun)
    bulan = int(bulan)
    hari = int(hari)
    jumlah_hari = calendar.monthrange(tahun, bulan)[1]
    n_pred = 1
    last = dataset_test.tail(1)
    fitur = last.values
    n_fit = len(fitur[0])
    fitur = fitur[:, 1:n_fit]
    y_pred=svr.predict(fitur)
    tahuns = np.zeros(n_pred)
    preds = np.array(y_pred)
    hari += 1
    if hari > jumlah_hari:
        bulan += 1
        hari = 1
    if bulan > 12:
      tahun += 1
      bulan = 1

    tanggal = str(tahun)+"-"+f"{bulan:02d}"+"-"+f"{hari:02d}"

    reshaped_data = preds.reshape(-1, 1)
    original_data = scaler.inverse_transform(reshaped_data)
    pred = original_data.flatten()
    # print(original_data)
    df_pred = pd.DataFrame({'Date': tanggal, 'Open': pred})

    # Plot data df



    # fig, ax = plt.subplots()
    # # Plot data df
    # ax.plot(df['Date'], df['Open'], label='Data Awal')

    # # Plot data df_pred
    # ax.plot(df_pred['Date'], df_pred['Open'], label='Prediksi')

    # # Menghubungkan plot terakhir data awal dengan plot awal data prediksi
    # last_year = df['Date'].iloc[-1]
    # ax.plot([last_year, df_pred['Date'].iloc[0]], [df['Open'].iloc[-1], df_pred['Open'].iloc[0]], 'k--')

    # # Konfigurasi plot
    # ax.set_xlabel('Date')
    # ax.set_ylabel('Open')
    # ax.set_title('Perbandingan Data Awal dan Prediksi')
    # ax.legend()

    # ax.xticks(rotation=45, ha='right')
    # ax.gca().xaxis.set_major_locator(mdates.MonthLocator())

    # Tampilkan plot
    # st.pyplot(fig)
    with col1:
        st.dataframe(df_pred)
    with col2:
        plt.plot(df['Date'], df['Open'], label='Data Awal')

        # Plot data df_pred
        plt.plot(df_pred['Date'], df_pred['Open'], label='Prediksi')

        # Menghubungkan plot terakhir data awal dengan plot awal data prediksi
        last_year = df['Date'].iloc[-1]
        plt.plot([last_year, df_pred['Date'].iloc[0]], [df['Open'].iloc[-1], df_pred['Open'].iloc[0]], 'k--')

        # Konfigurasi plot
        plt.xlabel('Date')
        plt.ylabel('Open')
        plt.title('Perbandingan Data Awal dan Prediksi')
        plt.legend()

        plt.xticks(rotation=45, ha='right')
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator())

        # Tampilkan plot
        # plt.show()
        st.pyplot(plt)