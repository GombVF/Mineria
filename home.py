import streamlit as st
from carga import upld
from EDA import EDA
from Seleccion import sel
from Clustering import cluster
from procla import clasificacion, pronostico


def borrarData():
    st.session_state.data=[]

class Paginas:
    def __init__(self):
        self.paginas=[]
    def add(self,titulo,funcion):
        self.paginas.append({'titulo':titulo,'funcion':funcion})
    
    def run(self):
        page=st.sidebar.selectbox('Funcionalidades',self.paginas,format_func=lambda pagina:pagina['titulo'])
        st.sidebar.button('Limpiar datos',on_click=borrarData)
        page['funcion']()
        #st.sidebar.button("Borrar datos",on_click=delData)

def main():
    st.set_option('deprecation.showPyplotGlobalUse', False)
    if 'data' not in st.session_state or len(st.session_state.data)==0:
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