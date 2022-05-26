import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image


def upld():
    img=Image.open('logo.jpeg')
    st.title("MCart")
    st.image(img,use_column_width=False)
    st.subheader("Esta es una herramienta de mineria de datos que funciona con archivos csv que te permite"
        +" analizar tus datos con multiples funcionalidades, es una herramienta gratuita y fácil de usar.")
    datos=st.file_uploader(label='Sube un archivo csv',type='csv')
    if datos:
        with st.form('opciones'):
            data=pd.read_csv(datos)
            st.session_state.data=[]
            st.session_state.data.append(data)
            st.session_state.sp=-1
            st.form_submit_button('Enviar')

        
    
        
        