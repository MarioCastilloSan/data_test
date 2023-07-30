import streamlit as st
import pandas as pd
from reg import newPrediction,readData
data = readData()
def main():
    st.title('Modelo de Predicción de Compras')

    # Interacción con el usuario
    st.header('Interacción con el usuario')

    # Aquí puedes agregar campos de entrada para que el usuario proporcione datos
    age = st.number_input('Edad', min_value=18, max_value=100, value=30)
    estimated_salary = st.number_input('Salario estimado', min_value=0, max_value=200000, value=50000)

    # Realizar predicción con los datos proporcionados por el usuario
    user_data = pd.DataFrame({
        'Age': [age],
        'EstimatedSalary': [estimated_salary]
    })

    # Cargar el modelo y realizar la predicción
    dataToPred = [[user_data['Age'].iloc[0],user_data['EstimatedSalary'].iloc[0]]]

    y_pred_user = newPrediction( dataToPred)
    print(y_pred_user)
    # Mostrar el resultado de la predicción para los datos del usuario
    st.subheader('Resultado de la predicción:')
    if y_pred_user[0] == 1:
        st.write('Es probable que el cliente compre el producto.')
    else:
        st.write('Es probable que el cliente NO compre el producto.')

    # Mostrar información relevante del conjunto de datos original
    st.header('Información relevante del conjunto de datos original:')
    st.write(data)


if __name__ == '__main__':
    main()
