import streamlit as st  
import pandas as pd
import numpy as np
import pickle
import sklearn
#configuracion de la pagina Prediccion Inversion tienda de video juegos
st.set_page_config(page_title="Predicción inversion tienda de video juego", page_icon=":guardsman:", layout="wide")

# Crear punto de entrada def main
def main():

    #Cargar imagen 
    st.image("video juego.jpg", width=900)

    #Cargar el modelo
    filename = 'modelo-reg-tree. RF.pkl'
    model_Tree,model_RF,variables = pickle.load(open(filename, 'rb')) #Cargar los modelos Tree,  RF y el objeto variebles




    #crear el sidebar de variables
    st.sidebar.title("Parámetros del usuario") # crear sibdar  

    # Crear los Campos de entrada para las variables
    def user_imput_features():
        edad = st.sidebar.number_input("Edad", min_value=14, max_value=52 ) # Edad del cliente
        option = ["'Mass Effect'","'Sim City'","'Crysis'","'Dead Space'","'Battlefield'","'KOA: Reckoning'","'F1'","'Fifa'"]
        videojuego = st.sidebar.selectbox("Videojuego", option, index=0)

        option_plataforma = ["'Play Station'","'Xbox'","PC","Otros"]
        plataforma = st.sidebar.selectbox("Plataforma", option_plataforma, index=0)

        option_sex = ["Hombre","Mujer"]
        sexo = st.sidebar.selectbox("sexo",option_sex, index=0)

        Consumidor_habitual = st.sidebar.checkbox("Consumidor habitual", value=False) # Si el cliente es un consumidor habitual o no

        # Crear un diccionario de las variables
        data={
            'Edad': edad,
            'Videojuego': videojuego,
            'Plataforma': plataforma,
            'Sexo': sexo,
            'Consumidor_habitual': Consumidor_habitual
        }
       # st.write(data) # Mostrar el diccionario de las variables
        data_imput = pd.DataFrame(data, index=[0]) # Crear un dataframe de las variables
        #st.write(data_imput) # Mostrar el dataframe de las variables

        return data_imput
    data_imp = user_imput_features()
    data_preparada = data_imp.copy() # Crear una copia del dataframe de las variables
    #st.write(data_preparada) # Mostrar el dataframe de las variables


    # Transfomar las variables categoricas en Dummies
    data_preparada = pd.get_dummies(data_preparada, columns=['Videojuego','Plataforma',], drop_first=False) # Transformar las variables categoricas en Dummies
    data_preparada = pd.get_dummies(data_preparada, columns=['Sexo'], drop_first=False) # Transformar las variables categoricas en Dummies
    #st.write(data_preparada) # Mostrar el dataframe de las variables transformadas

    # Ajustar el datafarme datos faltantes a la forma del modelo Reindexacion de columnas faltantes 
    data_preparada = data_preparada.reindex(columns=variables, fill_value=0) # Ajustar el dataframe a la forma del modelo
    #st.write(data_preparada) # Mostrar el dataframe de las variables ajustadas

    # Predecir el modelo de arbol de decision
    #crear el boton de prediccion
    if st.sidebar.button("Predecir"):
       y_pred_Tree = model_Tree.predict(data_preparada) #Predicción del modelo Tree
       #st.write(y_pred_Tree) # Mostrar la predicción del modelo Tree
        
       st.success(f" El cliente invertirá: {y_pred_Tree[0]: .1f} dolares") #Mostrar la predicción del modelo Tree
       st.write("Predicción del modelo Tree: 96%")
       
  
      
if __name__ == "__main__":
     main()

