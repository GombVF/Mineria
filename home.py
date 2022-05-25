from asyncio.windows_events import NULL
from dataclasses import dataclass
from turtle import onclick
import streamlit as st
import pandas as pd
import numpy as np
from carga import upld
from EDA import EDA
from Seleccion import sel
from Clustering import cluster
from procla import clasificacion, pronostico
import os

from procla import pronostico

def delData():
    os.remove('data/data.csv')

class Paginas:
    def __init__(self):
        self.paginas=[]
    def add(self,titulo,funcion):
        self.paginas.append({'titulo':titulo,'funcion':funcion})
    """def delData():
        os.remove('data/data.csv')
    def saveData(data):
        data.to_csv("data/data.csv")"""
    def run(self):
        page=st.sidebar.selectbox('Funcionalidades',self.paginas,format_func=lambda pagina:pagina['titulo'])
        page['funcion']()
        #st.sidebar.button("Borrar datos",on_click=delData)

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if 'data' not in st.session_state:
        upld()
    else:
        web= Paginas()  
        #web.add('Carga',upld)
        web.add('Analisis exploratorio',EDA)
        web.add('Selección de características',sel)
        web.add('Clustering',cluster)
        web.add('Pronóstico',pronostico)
        web.add('Clasificación',clasificacion)
        web.run()

if __name__=='__main__':
    main()