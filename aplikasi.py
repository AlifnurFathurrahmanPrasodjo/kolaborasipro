# import libary 
import streamlit as st
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from datetime import datetime
import matplotlib.dates as mdates
import calendar

# pige title
st.set_page_config(
    page_title="Forecasting Harga Saham PT. INDOFOOD (Time Series Data)",
    page_icon="https://abisgajian.id/images/thumbnail/ini-dia-daftar-saham-kategori-blue-chip-di-bursa-saham-indonesia.jpg",
)

    # 0 = Anda Tidak Depresi
    # 1 = Anda Depresi

# hide menu
hide_streamlit_style = """

<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

# insialisasi web
st.markdown("<h1 style='text-align: center; color: white; margin:0 ; padding:0;'>Forecasting Harga Saham PT. INDOFOOD (Time Series Data)</h1>", unsafe_allow_html=True)
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">', unsafe_allow_html=True)
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.markdown(' <div style="position: fixed; top: 0; left: 0; z-index: 9999; width: 100%; background: rgb(14, 17, 23); ; text-align: center;"><a href="https://github.com/AlifnurFathurrahmanPrasodjo/Project-Penambangan-Data.git" target="_blank"><button style="border-radius: 12px;position: relative; top:50%; margin:10px;"><i class="fa fa-github"></i> GitHub</button></a><a href="https://alifnurfathurrahmanprasodjo.github.io/DATAMINING/project_pendat.html?highlight=project" target="_blank"><button  style="border-radius: 12px;position: relative; top:50%;"><i style="color: orange" class="fa fa-book"></i> Jupyter Book</button></a></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Data", "Preprocessing Data", "Model", "Implementasi"])
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.write("Deskripsi Aplikasi :")
        st.markdown("<p style='text-align: justify;'>Aplikasi Peramalan Data Saham PT INDOFOOD adalah perangkat lunak yang dirancang untuk memberikan kemampuan peramalan dan analisis untuk data saham PT INDOFOOD, sebuah perusahaan fiktif. Aplikasi ini bertujuan untuk membantu investor, pedagang, dan analis keuangan dalam membuat keputusan dengan memprediksi harga dan tren saham di masa depan.</p>", unsafe_allow_html=True)
        st.write("Sumber data :")
        st.markdown("<p style='text-align: justify;'>Aplikasi mengambil data stok historis PT INDOFOOD dari sumber terpercaya https://finance.yahoo.com/quote/INDF.JK/profile?p=INDF.JK. Ini mengambil data seperti harga saham harian, volume perdagangan, dan metrik keuangan relevan lainnya.</p>", unsafe_allow_html=True)
        st.write("Deskripsi Data :")
        st.write("1. Date: Tanggal entri data pasar saham.\n 2. Open: Harga pembukaan saham pada hari itu.\n 3. High: Harga tertinggi yang dicapai saham selama hari perdagangan.\n 4. Low: Harga terendah yang dicapai oleh saham selama hari perdagangan.\n 5. Close: Harga penutupan saham pada hari itu.\n 6. Adj Close: Harga penutupan saham yang disesuaikan, yang memperhitungkan tindakan korporasi apa pun (seperti dividen atau pemecahan saham) yang dapat memengaruhi harga saham.\n 7. Volume: Volume perdagangan, yaitu jumlah total saham yang diperdagangkan pada hari itu.")
    with col2:
        data = pd.read_csv('https://raw.githubusercontent.com/AlifnurFathurrahmanPrasodjo/dataFolder/main/dataMining/INDF.JK.csv')
        data

with tab2:
    df = pd.read_csv('https://raw.githubusercontent.com/AlifnurFathurrahmanPrasodjo/kolaborasipro/master/DatasetSaham.csv')
    df  

    dataSplit='''
    import joblib
data = df['Open']
n = len(data)
sizeTrain = (round(n*0.8))
data_train = pd.DataFrame(data[:sizeTrain])
data_test = pd.DataFrame(data[sizeTrain:])
data_train
    '''
    st.code(dataSplit, language='python')

    import joblib
    data = df['Open']
    n = len(data)
    sizeTrain = (round(n*0.8))
    data_train = pd.DataFrame(data[:sizeTrain])
    data_test = pd.DataFrame(data[sizeTrain:])
    data_train

    normalisasiData='''
    from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(data_train)

# Mengaplikasikan MinMaxScaler pada data pengujian
test_scaled = scaler.transform(data_test)

# reshaped_data = data.reshape(-1, 1)
train = pd.DataFrame(train_scaled, columns = ['data'])
train = train['data']

test = pd.DataFrame(test_scaled, columns = ['data'])
test = test['data']
joblib.dump(scaler, 'scalerUAS.pkl')
test
    '''
    st.code(normalisasiData, language='python')

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(data_train)

    # Mengaplikasikan MinMaxScaler pada data pengujian
    test_scaled = scaler.transform(data_test)

    # reshaped_data = data.reshape(-1, 1)
    train = pd.DataFrame(train_scaled, columns = ['data'])
    train = train['data']

    test = pd.DataFrame(test_scaled, columns = ['data'])
    test = test['data']
    # joblib.dump(scaler, 'scalerUAS.pkl')
    test    

    ekstraksiFitur='''
    import numpy as np
from numpy import array
def split_sequence(sequence, n_steps):
X, y = list(), list()
for i in range(len(sequence)):
# find the end of this pattern
    end_ix = i + n_steps
    # check if we are beyond the sequence
    if end_ix > len(sequence)-1:
    break
    # gather input and output parts of the pattern
    seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
    X.append(seq_x)
    y.append(seq_y)

return array(X), array(y)
df_X, df_Y = split_sequence(train, 2)
x = pd.DataFrame(df_X)
y = pd.DataFrame(df_Y)
dataset_train = pd.concat([x, y], axis=1)
dataset_train
# dataset_train.to_excel('data-train.xlsx', index=False)
X_train = dataset_train.iloc[:, :2].values
Y_train = dataset_train.iloc[:, -1].values
test_x, test_y = split_sequence(test, 2)
x = pd.DataFrame(test_x)
y = pd.DataFrame(test_y)
dataset_test = pd.concat([x, y], axis=1)
X_test = dataset_test.iloc[:, :2].values
Y_test = dataset_test.iloc[:, -1].values
dataset_test
    '''    
    st.code(ekstraksiFitur, language='python')

    import numpy as np
    from numpy import array
    def split_sequence(sequence, n_steps):
        X, y = list(), list()
        for i in range(len(sequence)):
            # find the end of this pattern
            end_ix = i + n_steps
            # check if we are beyond the sequence
            if end_ix > len(sequence)-1:
                break
            # gather input and output parts of the pattern
            seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
            X.append(seq_x)
            y.append(seq_y)
        
        return array(X), array(y)
    df_X, df_Y = split_sequence(train, 3)
    x = pd.DataFrame(df_X, columns=['t-3','t-2','t-1'])
    y = pd.DataFrame(df_Y, columns=['t'])
    dataset_train = pd.concat([x, y], axis=1)
    dataset_train = dataset_train.reset_index(drop=True)
    dataset_train
    # dataset_train.to_excel('data-train.xlsx', index=False)
    X_train = dataset_train.iloc[:, :3].values
    Y_train = dataset_train.iloc[:, -1].values
    test_x, test_y = split_sequence(test, 3)
    x = pd.DataFrame(test_x, columns=['t-3','t-2','t-1'])
    y = pd.DataFrame(test_y, columns=['t'])
    dataset_test = pd.concat([x, y], axis=1)
    dataset_test = dataset_test.reset_index(drop=True)
    X_test = dataset_test.iloc[:, :3].values
    Y_test = dataset_test.iloc[:, -1].values
    dataset_test

with tab3:
    modellingData='''
    from sklearn.svm import SVR
    from sklearn.metrics import mean_absolute_percentage_error
    import matplotlib.pyplot as plt

    models = []
    uji = ['linear','rbf','poly']
    errors = []
    for ker in uji:

        svr = SVR(kernel=ker) 
        svr.fit(X_train, Y_train)
        
        y_pred=svr.predict(X_test)
        error = mean_absolute_percentage_error(y_pred, Y_test)
        models.append(svr)
        errors.append(error)
        
        plt.plot(y_pred, label='Prediksi')
        plt.plot(Y_test, label='Aktual')
        plt.xlabel('Index')
        plt.ylabel('Nilai')
        plt.title('Plot Prediksi vs Aktual')
        plt.legend()
        plt.show()

    indexModel = np.argmin(errors)
    joblib.dump(models[indexModel], 'modelSVM.pkl')
    '''
    st.code(modellingData, language='python')

    from sklearn.svm import SVR
    from sklearn.metrics import mean_absolute_percentage_error
    import matplotlib.pyplot as plt

    models = []
    uji = ['linear','rbf','poly']
    errors = []
    for ker in uji:

        svr = SVR(kernel=ker) 
        svr.fit(X_train, Y_train)
        
        y_pred=svr.predict(X_test)
        error = mean_absolute_percentage_error(y_pred, Y_test)
        models.append(svr)
        errors.append(error)
        
        plt.plot(y_pred, label='Prediksi')
        plt.plot(Y_test, label='Aktual')
        plt.xlabel('Index')
        plt.ylabel('Nilai')
        plt.title('Plot Prediksi vs Aktual')
        plt.legend()
        st.pyplot(plt)

        st.write("MAPE = ",error)
    

with tab4:
    # Mengimpor model dan text dari file yang telah diekspor sebelumnya
    svr = joblib.load('modelSVM.pkl')
    df = pd.read_csv('DatasetSaham.csv')
    scaler = joblib.load('scalerUAS.pkl')
    dataset_test = pd.read_csv('data-test-UAS.csv')


    # Judul aplikasi
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
        # input1 = fitur[0]
        # input2 = fitur[1]
        # input3 = fitur [2]
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

        with col1:
            last_three_rows = df['Open'].tail(3).values
            # st.dataframe(last_three_rows)
            # st.write('input = ',df['Open'].tail.values)
            st.write('input t-2 = ',last_three_rows[0])
            st.write('input t-2 = ',last_three_rows[1])
            st.write('input t-1 = ',last_three_rows[2])
            st.write('hasil prediksi = ',pred[0])
        with col2:
           pass
