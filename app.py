import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# Cargar el modelo y el escalador previamente guardados
scaler = joblib.load('scaler.bin')
knn_model = joblib.load('knn_model.bin')

# Título y subtítulo
st.title('Predicción de problemas cardiacos')
st.subheader('Realizado por Leonardo Waked')

# Instrucción de manejo y objetivo
st.write('Este sistema predice si existe riesgo de problemas cardiacos basado en los parámetros proporcionados.')
st.write('Ajusta los controles deslizantes para ingresar tu edad y nivel de colesterol, luego presiona el botón para predecir.')

# Mostrar la imagen
st.image('https://images.emojiterra.com/google/noto-emoji/unicode-15/color/512px/1fac0.png', width=200)

# Controles deslizantes para la entrada de los datos
edad = st.slider('Edad', min_value=20, max_value=80, value=40)
colesterol = st.slider('Colesterol', min_value=100, max_value=600, value=200)

# Crear un dataframe con los valores introducidos
input_data = pd.DataFrame([[edad, colesterol]], columns=['edad', 'colesterol'])

# Escalar los datos de entrada utilizando el scaler cargado
input_scaled = scaler.transform(input_data)

# Botón para hacer la predicción
if st.button('Predecir'):
    # Realizar la predicción utilizando el modelo KNN
    prediccion = knn_model.predict(input_scaled)

    # Mostrar el resultado
    if prediccion == 0:
        st.markdown('<div style="background-color:green; padding:10px; color:white;">No tiene riesgo de problemas cardiacos</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background-color:rgba(255, 0, 0, 0.8); padding:10px; color:white;">Tiene riesgo de problemas cardiacos</div>', unsafe_allow_html=True)
    
    st.write(f'Valor predicho: {prediccion[0]}')

# Línea de separación y mensaje de copyright
st.markdown('---')
st.markdown('**Copyright © UNAB 2025**')
