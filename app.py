import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from reg import newPrediction,readData,getDatatoPlot
data = readData()

def plot_decision_boundary(model, x, y, title):
    # Código para generar el gráfico de dispersión y la frontera de decisión sin cambios
    xSet, ySet = x, y
    xSet_2d = np.column_stack((xSet[:, 0], xSet[:, 1]))  # Convertir a arreglo bidimensional

    x1, x2 = np.meshgrid(np.arange(start=xSet_2d[:, 0].min() - 1, stop=xSet_2d[:, 0].max() + 1, step=0.01),
                         np.arange(start=xSet_2d[:, 1].min() - 1, stop=xSet_2d[:, 1].max() + 1, step=0.01))
    Z = model.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)

    fig, ax = plt.subplots()
    ax.contourf(x1, x2, Z, alpha=0.75, cmap=ListedColormap(('red', 'green')))
    ax.set_xlim(x1.min(), x1.max())
    ax.set_ylim(x2.min(), x2.max())

    for i, j in enumerate(np.unique(ySet)):
        ax.scatter(xSet[ySet == j, 0], xSet[ySet == j, 1],
                   c=ListedColormap(['red', 'green'])(i), label=j)

    plt.title(title)
    plt.xlabel("Age")
    plt.ylabel("Estimated Salary")
    plt.legend()

    st.pyplot(fig)

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

    # Mostrar el gráfico de la frontera de decisión
    xTrain,yTrain,naiveBayesModel = getDatatoPlot()
    # Agregar controles deslizantes para ajustar los valores de Age y EstimatedSalary
    age = st.slider('Edad', min_value=int(xTrain[:, 0].min()), max_value=int(xTrain[:, 0].max()), value=int(xTrain[:, 0].mean()))
    estimated_salary = st.slider('Salario estimado', min_value=int(xTrain[:, 1].min()), max_value=int(xTrain[:, 1].max()), value=int(xTrain[:, 1].mean()))

    # Mostrar el gráfico de la frontera de decisión con interactividad
    plot_decision_boundary(naiveBayesModel, xTrain, yTrain, "Naive Bayes Classifier (Training Set)")


if __name__ == '__main__':
    main()
