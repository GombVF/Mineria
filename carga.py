import streamlit as st
import pandas as pd
import numpy as np
#from home import main


def upld():
    st.title("Bienvenido.")
    st.subheader("Esta es una herramienta de mineria de datos que funciona con archivos csv que te permite"
        +"analizar tus datos con multiples funcionalidades, es una herramienta gratuita y fácil de usar.")
    datos=st.file_uploader(label='Sube un archivo csv',type='csv')
    if datos:
        with st.form('opciones'):
            h=st.radio("Cabecera",['Sí','No'])
            if h=='Sí':
                data=pd.read_csv(datos)
            else:
                data=pd.read_csv(datos,header=None)
            st.session_state.data=[]
            st.session_state.data.append(data)
            st.session_state.sp=-1
            st.form_submit_button('Enviar')

        
    
        
        