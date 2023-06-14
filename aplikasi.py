import streamlit as st





# Judul aplikasi
st.markdown("<h1 style='text-align: center;'>Forecasting Temperature Anomalies Data Time Series</h1>", unsafe_allow_html=True)
# st.title('Forecasting Temperature Anomalies Data Time Series')

# Define a range for the slider
min_value = 0
max_value = 60

# Create a slider widget
selected_value = st.slider('Select a value', min_value, max_value)

# Display the selected value
st.write('You selected:', selected_value)

# Tombol untuk memprediksi
if st.button('Run'):
    if selected_value:
        st.write(selected_value)
    else:
        st.write('Masukkan range tahun terlebih dahulu')