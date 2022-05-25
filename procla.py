from scipy import rand
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, max_error, r2_score 

def reg(**kwargs):
    return linear_model.LinearRegression()
def tree(**kwargs):
    return DecisionTreeRegressor(max_depth=kwargs['depth'], min_samples_split=kwargs['split'], min_samples_leaf=kwargs['leafs'], random_state=kwargs['random'])
def forest(**kwargs):
    return RandomForestRegressor(n_estimators=kwargs['estimators'], max_depth=kwargs['depth'], min_samples_split=kwargs['split'], min_samples_leaf=kwargs['leafs'], random_state=kwargs['random'])

def log(**kwargs):
    return linear_model.LogisticRegression()
def treec(**kwargs):
    return DecisionTreeClassifier()
def forec(**kwargs):
    return RandomForestClassifier()

def pronostico():
    st.title("Predicción.")
    st.session_state.aux= 'aux'
    model_dict={'Regresión lineal':reg,'Arboles':tree,'Bosque':forest}
    data=st.session_state.data[st.session_state.sp]
    est=''
    d=''
    split=''
    leafs=''
    with st.container():
        with st.form('Modelo de predicción'):
            predict_type=st.radio('Modelo de predicción',['Regresión lineal','Arboles','Bosque'])
            sb=st.form_submit_button('Continuar.')
        with st.form('Opciones de predicción.'):
            dependant_var=st.selectbox('Seleccionar variable a ser predicha.',data.columns)
            st.dataframe(data[dependant_var])
            if predict_type=='Arboles':
                d=st.number_input('Profundidad maxima',2)
                split=st.number_input('Cantidad mínima para hacer split',2)
                leafs=st.number_input('Cantidad minima de hojas',1)
            elif predict_type=='Bosque':
                est=st.number_input('Número de estimadores',1)
                d=st.number_input('Profundidad maxima',2)
                split=st.number_input('Cantidad mínima para hacer split',2)
                leafs=st.number_input('Cantidad minima de hojas',1)
            sb=st.form_submit_button('Predecir')
        if sb:
            independant_var=data.columns.drop(dependant_var)
            X= np.array(data[independant_var])
            Y= np.array(data[dependant_var])
            x_train,x_test,y_train,y_test= model_selection.train_test_split(X,Y,test_size=0.2,random_state=1234,shuffle=True)
            prediccion=model_dict[predict_type](estimators=est,depth=d,split=split,leafs=leafs,random=1234)
            prediccion.fit(x_train,y_train)
            Yp=prediccion.predict(x_test)
            plt.plot(y_test,color='green',marker='o',label='Datos de prueba')
            plt.plot(Yp,color='red',marker='o',label='Datos de calculados')
            fig=plt.plot()
            st.pyplot(fig)
            if predict_type=='Regresión lineal':
                st.text('Coeficientes: '+str(prediccion.coef_))
                st.text('Intercepto: '+str(prediccion.intercept_))
                st.text('Residuo: '+str(max_error(y_test,Yp)))
                st.text('MSE: '+str(mean_squared_error(y_test,Yp)))
                st.text('RMSE: '+str(mean_squared_error(y_test,Yp,squared=False)))
                st.text('Score: '+str(r2_score(y_test,Yp)))
            else:
                st.text('Criterio: '+str(prediccion.criterion))
                st.text('Peso o importancia: '+str(prediccion.feature_importances_))
                st.text('MAE: '+str(mean_absolute_error(y_test,Yp)))
                st.text('MSE: '+str(mean_squared_error(y_test,Yp)))
                st.text('RMSE: '+str(mean_squared_error(y_test,Yp,squared=False)))
                st.text('Score: '+str(r2_score(y_test,Yp)))
            


def clasificacion():
    st.title("Clasificación.")
    st.session_state.aux= 'aux'
    model_dict={'Regresión logística':log,'Arboles':treec,'Bosque':forec}
    data=st.session_state.data[st.session_state.sp]
    est=''
    d=''
    split=''
    leafs=''
    with st.container():
        with st.form('Modelo de predicción'):
            predict_type=st.radio('Modelo de predicción',['Regresión logística','Arboles','Bosque'])
            sb=st.form_submit_button('Continuar.')
        with st.form('Opciones de predicción.'):
            dependant_var=st.selectbox('Seleccionar variable a ser predicha.',data.columns)
            st.dataframe(data[dependant_var])
            if predict_type=='Arboles':
                d=st.number_input('Profundidad maxima',2)
                split=st.number_input('Cantidad mínima para hacer split',2)
                leafs=st.number_input('Cantidad minima de hojas',1)
            elif predict_type=='Bosque':
                est=st.number_input('Número de estimadores',1)
                d=st.number_input('Profundidad maxima',2)
                split=st.number_input('Cantidad mínima para hacer split',2)
                leafs=st.number_input('Cantidad minima de hojas',1)
            sb=st.form_submit_button('Predecir')
        if sb:
            independant_var=data.columns.drop(dependant_var)
            X= np.array(data[independant_var])
            Y= np.array(data[dependant_var])
            x_train,x_validation,y_train,y_validation= model_selection.train_test_split(X,Y,test_size=0.2,random_state=1234,shuffle=True)
            clasificador=model_dict[predict_type](estimators=est,depth=d,split=split,leafs=leafs,random=1234)
            clasificador.fit(x_train,y_train)
            Yp=clasificador.predict(x_validation)
            st.text('Matriz de confusión: ')
            st.dataframe(pd.crosstab(y_validation.ravel(),Yp,rownames=['Real'],colnames=['Clasificación']))
            if predict_type=='Regresión logística':
                st.text('Exactitud: '+str(clasificador.score(x_validation,y_validation)))
                st.text('Información general:')
                st.dataframe(pd.DataFrame(classification_report(y_validation,Yp,output_dict=True)).transpose())
                st.text('Intercepción: '+str(clasificador.intercept_))
                st.text('Coeficientes: '+str(clasificador.coef_))
            else:
                st.text('Criterio: '+str(clasificador.criterion))
                st.text('Exactitud: '+str(clasificador.score(x_validation,y_validation)))
                st.text('Información general:')
                st.dataframe(pd.DataFrame(classification_report(y_validation,Yp,output_dict=True)).transpose())
                st.dataframe(pd.DataFrame({'Variable':list(data[independant_var]),'Peso':clasificador.feature_importances_}).sort_values('Peso',ascending=False))