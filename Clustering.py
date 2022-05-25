import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


def jera(**kwargs):
    with st.form('Clusters'):
        num_cluster=st.number_input('Número de clusters: ',2,16)
        sb=st.form_submit_button('Aplicar')
    if sb:
        clust= AgglomerativeClustering(n_clusters=num_cluster,linkage='complete',affinity=kwargs['medida'])
        clust.fit_predict(kwargs['dataE'])
        kwargs['data']['Cluster']= clust.labels_
        st.text('Clusters creados')
        st.dataframe(kwargs['data'].groupby(['Cluster'])['Cluster'].count())
        st.text('Información de los cluster')
        st.dataframe(kwargs['data'].groupby('Cluster').mean())
        st.text('Nuevos datos creados.')
        st.dataframe(kwargs['data'])


def part(**kwargs):
    with st.form('Clusters'):
        num_cluster=st.number_input('Número de clusters: ',2,16)
        sb=st.form_submit_button('Aplicar')
    if sb:
        clust=KMeans(n_clusters=num_cluster,random_state=0).fit(kwargs['dataE'])
        clust.predict(kwargs['dataE'])
        kwargs['data']['Cluster']= clust.labels_
        st.text('Clusters creados')
        st.dataframe(kwargs['data'].groupby(['Cluster'])['Cluster'].count())
        st.text('Información de los cluster')
        st.dataframe(kwargs['data'].groupby('Cluster').mean())
        st.text('Nuevos datos creados.')
        st.dataframe(kwargs['data'])

def auxMethod(**kwargs):
    return st.text('Esperando opciones de cluster...')

def cluster():
    st.title("Clustering.")
    method_dict={'Normalizado':StandardScaler(),'Escalado':MinMaxScaler()}
    medida_dict={'Euclidiana':'euclidean','Chebyshev':'shebyshev','Manhattan':'cityblock','Minkowski':'minkowski'}
    clust_dict={'Jerarquico':jera,'Particional':part,'aux':auxMethod}
    data= st.session_state.data[st.session_state.sp]
    with st.container():
        with st.form('Opciones del cluster'):
            method=st.radio('Tipo de estandarizado',['Normalizado','Escalado'])
            cluster_type=st.radio('Tipo de clusterizado',['Jerarquico','Particional'])
            medida=st.radio('Tipo de medición',['Euclidiana','Chebyshev','Manhattan','Minkowski'])
            sb=st.form_submit_button('Calcular')
        dataE=method_dict[method].fit_transform(data)
        SSE=[]
        if sb and cluster_type=='Jerarquico':
            shc.dendrogram(shc.linkage(dataE,method='complete',metric=medida_dict[medida]))
            fig=plt.plot()
            st.pyplot(fig)
            st.session_state.aux=cluster_type
        elif sb:
            st.session_state.aux=cluster_type
            for i in range(2,16):
                km=KMeans(n_clusters=i,random_state=0)
                km.fit(dataE)
                SSE.append(km.inertia_)
            plt.xlabel('Cantidad de clusters')
            plt.ylabel('SSE')
            plt.plot(range(2,16),SSE,marker='o')
            fig=plt.plot()
            st.pyplot(fig)
            k= KneeLocator(range(2,16),SSE,curve='convex',direction='decreasing')
            plt.style.use('ggplot')
            st.pyplot(k.plot_knee())
            st.text('Se recomiendan '+str(k.elbow)+' cluster.')  
    with st.container():
        dataE=method_dict[method].fit_transform(data)
        clust_dict[st.session_state.aux](data=data,dataE=dataE,medida=medida_dict[medida])


            