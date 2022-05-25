from pkgutil import extend_path
from scipy.fftpack import ss_diff
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def heat_map(data):
    hm=np.triu(data.corr())
    sns.heatmap(data.corr(),cmap='RdBu_r',annot=True,mask=hm)
    fig=plt.plot()
    st.pyplot(fig)

def componentes(estandar,data):
    cols=data.select_dtypes(include='object')
    data_aux=data.drop(columns=cols)
    dataE=estandar.fit_transform(data_aux)
    pca=PCA(n_components=len(data_aux.columns))
    pca.fit(dataE)
    var=pca.explained_variance_ratio_
    plt.xlabel('Número de componentes')
    plt.ylabel('Varianza acumulada')
    plt.plot(np.cumsum(var))
    plt.grid()
    fig=plt.plot()
    for i,j in enumerate(var):
        if sum(var[0:i])<0.9 and sum(var[0:i])>0.75:
            st.text('Se consigue un '+str(float(sum(var[0:i])))+' de varianza acumulada con '+str((i+1))+' componentes')
    st.pyplot(fig)
    
    st.dataframe(pd.DataFrame(abs(pca.components_),columns=data_aux.columns))


def sel():
    st.session_state.aux= 'aux'
    st.title("Selección de características.")
    data=st.session_state.data[st.session_state.sp]
    with st.container():
        with st.form('Opciones'):
            selection=st.radio('Tipo de selección: ',['Correlaciones','Componentes principales'])
            method=st.radio('Tipo de estandarizado',['Normalizado','Escalado'])
            sb=st.form_submit_button('Calcular')
        if sb:
            if selection=='Componentes principales':
                if method=='Normalizado':
                    estandar= StandardScaler()
                    componentes(estandar,data)
                else:
                    estandar= MinMaxScaler()
                    componentes(estandar,data)
            else:
                st.text('Datos con correlación.')
                st.dataframe(data.corr(method='pearson'))
                st.text('Datos categóricos.')
                st.dataframe(data.select_dtypes(include='object'))
                heat_map(data)
    with st.container():
        with st.form('Selección de variables.'):
            remove_vars=st.multiselect('Selecciona las variables a retirar',data.columns)
            sb=st.form_submit_button('Retirar')
        if len(remove_vars)!=0 and sb:
            st.session_state.data.append(data.drop(columns=remove_vars))
    with st.container():
        with st.form('Historial.'):
            st.text('Historial de datos')
            sb=st.form_submit_button('Mostrar')
        if sb:
            for i,j in enumerate(st.session_state.data):
                if i==0:
                    st.text('========= version: original (0) =========')
                else:
                    st.text('========= version: '+str(i)+'=========')
                st.dataframe(j)
        with st.form('Cambiar version.'):
            version=st.number_input('Ingresa el número de version que desea.',min_value=0)
            sb=st.form_submit_button('Enviar')
        if sb:
            st.session_state.data=st.session_state.data[:version+1]
            st.text('Version actual:')
            dataux=st.session_state.data[st.session_state.sp]
            st.dataframe(dataux)

    with st.container():
        with st.form('Convertir variables'):
            var=st.selectbox('Selecciona la variable cuyos valores quieres convertir',data.columns)
            st.text('Nota: para realizar el cambio de valores se debe de seguir el siguiente formato.')
            st.text('val1:newVal1,val2:newVal2')
            st.text('Donde val es el valor a cambiar, cada valor a cambiar debe de estar seguido por dos puntos y el nuevo valor.')
            st.text('Para cambiar más de un valor, es necesario separar los valores con comas')
            c=st.text_input('Ingresa el cambio')
            sb=st.form_submit_button('Cambiar')
        if sb:
            c=c.split(',')
            for i in c:
                aux=i.split(':')
                st.text(print(aux))
                data=data.replace({aux[0]:aux[1]})
            st.session_state.data.append(data)
            data=st.session_state.data[st.session_state.sp]
            st.dataframe(pd.value_counts(data[var]))
            st.text(pd.value_counts(data[var]))
        st.dataframe(data)

            
