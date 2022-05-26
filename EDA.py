import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io



def visualizador(data,variables, tipo,compare):
    if tipo=='Histrogramas':
        if not compare:
            for i in variables:
                plt.hist(data[i])
                plt.grid()
                plt.title(i)
                plt.ylabel('Cantidad de datos')
                plt.xlabel('Valor de los datos')
                fig=plt.plot(figsize=(50,50))
                st.pyplot(fig,clear_figure=True)
        else:
            data[variables].hist()
            fig=plt.plot()
            st.pyplot(fig)
    elif tipo=='Caja y bigote':
        if not compare:
            for i in variables:
                plt.boxplot(data[i])
                plt.xticks([1],[i])
                plt.ylabel('valor de los datos')
                plt.grid()
                fig=plt.plot()
                st.pyplot(fig)
        else:
            data[variables].boxplot()
            plt.ylabel('valor de los datos')
            fig=plt.plot()
            st.pyplot(fig)
    elif tipo=='Barra':
        for i in variables:
            sns.countplot(y=i,data=data)
            fig=plt.plot()
            st.pyplot(fig)
            


def EDA():
    st.session_state.aux= 'aux'
    st.title("Análisis exploratorio de datos.")
    st.sidebar.subheader('Información de sección.')
    st.sidebar.write('Es importante conocer el comportamiento de nuestros datos'
        +' es por eso que debemos de realizar un análisis sobre ellos.')
    st.sidebar.write('Es posible seleccionar las variables que se desean anaizar'
        +' además de poder compararlas en una misma gráfica.')
    st.sidebar.write('NOTA: Las gráficas de barra no pueden ser comparadas ya que'
        +' se pensaron para graficar variables categoricas y cada variable'
        +' tiene su propia gráfica.')
    data=st.session_state.data[st.session_state.sp]
    vars=[]
    with st.expander('Información general'):
        st.text('Información de los datos: ')
        buffer= io.StringIO()
        data.info(buf=buffer)
        s=buffer.getvalue()
        s=s.splitlines()
        s=s[3:]
        s='\n'.join(s)
        st.text(s)
        st.text('Información estadística: ')
        st.dataframe(data.describe())
    for i in data.columns:
        vars.append(i)
    with st.expander('Visualizar variables'):
        with st.form('graficas'):
            try:
                vars_to_show=st.multiselect('Variables',vars)
                graph=st.radio('Tipo de gráfica',['Histrogramas','Caja y bigote','Barra'])
                t_show=st.checkbox('Comparar')
                sb= st.form_submit_button('Ver')
                if sb and len(vars_to_show)==0:
                    st.title('Selecciona las variables a visualizar')
                elif sb:
                    visualizador(data,vars_to_show,graph,t_show)
            except:
                st.title('No es posible graficar esa(s) variable(s)')
    st.dataframe(data)
    