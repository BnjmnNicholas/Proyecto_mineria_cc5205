# Importación de librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
import umap.umap_ as umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from pandas.plotting import parallel_coordinates
import os
import shutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')



def scaler(df, scaler_method_name, numerical_cols):
    '''
    Transforma las variables numericas del dataframe según el scaler escogido.
    Parameters:
        df (pd.DataFrame): Dataframe a escalar.
        scaler_method_name (str): Nombre del scaler a utilizar.
        numerical_cols (list): Lista de columnas numericas a escalar.
    Returns:
        df_copy (pd.DataFrame): Dataframe con las columnas numericas escaladas.
    '''

    df_copy = df.copy()

    # segun el nombre del metodo, crea el objeto y ejecuta el escalado
    if scaler_method_name == 'StandardScaler':
        scaler = StandardScaler()
        df_copy[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df_copy
    elif scaler_method_name == 'PowerTransform':
        power = PowerTransformer(method = 'yeo-johnson', standardize = True)
        power.fit(df_copy[numerical_cols])
        df_copy[numerical_cols] = power.transform(df[numerical_cols])
        return df_copy

def encoder(df, encoder_method_name, categorical_cols):
    '''
    Transforma las variables categoricas del dataframe según el encoder escogido.
    '''
    if encoder_method_name == 'OneHotEncoder':
        df = pd.get_dummies(df, columns=categorical_cols)
        return df
    


def preprocess(df, scaler_method_name, numerical_cols, encoder_method_name, categorical_cols):

    categorical_cols = [col for col in df.columns if col not in numerical_cols]
    df = scaler(df, scaler_method_name = scaler_method_name, numerical_cols= numerical_cols)
    df = encoder(df, encoder_method_name = encoder_method_name, categorical_cols = categorical_cols)
    return df


def k_means_elbow(df, model_name, output_file_path):
    '''
    Utiliza el metodo de la rodilla para determinar el número óptimo de clusters.
    Parameters:
        df (pd.DataFrame): Dataframe a utilizar.
        model_name (str): Nombre del modelo a utilizar.
        output_file_path (str): Path donde se guardará la visualización.  
    Returns:
        optimal_k (int): Número óptimo de clusters.
    '''
    sse = []

    k_range = list(range(2, 11))
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=10).fit(df)
        sse.append(kmeans.inertia_)

    optimal_k = k_range[0]
    min_sse = float('inf')
    max_sse = float('-inf')
    for i, k in enumerate(k_range[3:10]):
        sse_before = sse[i-1]
        sse_after = sse[i+1]
        dif_before = sse_before - sse[i]
        dif_after = sse[i] - sse_after
        if dif_before > max_sse and dif_after < min_sse:
            min_sse = sse[i]
            max_sse = sse[i]
            optimal_k = k

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sse, marker='o')
    plt.xlabel('NÚMERO DE CLUSTERS (k)')
    plt.ylabel('INERTIA')
    plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
    plt.title(f'MÉTODO DEL CODO - {model_name}', fontsize=12)
    plt.legend()
    plt.savefig(output_file_path)
    plt.close()

    print('knee method visualization saved')
    return optimal_k

# Nueva version corregida
def K_means_silhouette(df, model_name, output_file_path):
    '''
    Visualiza el coeficiente de silueta para determinar el número óptimo de clusters.
    Parameters:
        df (pd.DataFrame): Dataframe a utilizar.
        model_name (str): Nombre del modelo a utilizar.
        output_file_path (str): Path donde se guardará la visualización.  
    Returns:
        optimal_k (int): Número óptimo de clusters.
    '''
    k_range = range(2, 11)
    sil = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df)
        score = silhouette_score(df, kmeans.labels_)
        sil.append(score)

    optimal_k = k_range[0]
    max_sil = float('-inf')
    for i, k in enumerate(k_range):
        if 2 <= k <= 8 and sil[i] > max_sil:
            max_sil = sil[i]
            optimal_k = k

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sil, marker='o')
    plt.xlabel('NÚMERO DE CLUSTERS (k)')
    plt.ylabel('COEFICIENTE DE SILUETA')
    plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
    plt.title(f'MÉTODO DE COEF. SILUETA - {model_name}', fontsize=12)
    plt.legend()
    plt.savefig(output_file_path)
    plt.close()

    print('Silhouette score visualization saved')
    return optimal_k

# Nueva version corregida
def Agglomerative_silhouette(df, model_name, output_file_path):
    """
    Visualiza el coeficiente de silueta para determinar el número óptimo de clusters.
    Parameters:
        df (pd.DataFrame): Dataframe a utilizar.
        model_name (str): Nombre del modelo a utilizar.
        output_file_path (str): Path donde se guardará la visualización.
    Returns:
        optimal_k (int): Número óptimo de clusters.
    """
    k_range = range(2, 11)
    sil = []
    for k in k_range:
        agglomerative = AgglomerativeClustering(n_clusters=k)
        labels = agglomerative.fit_predict(df)
        score = silhouette_score(df, labels)
        sil.append(score)

    optimal_k = k_range[0]
    max_sil = float('-inf')
    for i, k in enumerate(k_range):
        if 2 <= k <= 8 and sil[i] > max_sil:
            max_sil = sil[i]
            optimal_k = k

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sil, marker='o')
    plt.xlabel('NÚMERO DE CLUSTERS (k)')
    plt.ylabel('COEFICIENTE DE SILUETA')
    plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
    plt.title(f'MÉTODO DE COEF. SILUETA - {model_name}', fontsize=12)
    plt.legend()
    plt.savefig(output_file_path)
    plt.close()

    print('Silhouette score visualization saved')
    return optimal_k


# Nueva version corregida
def gaussian_mixture_silhouette(df, model_name, output_file_path):
    """
    Visualiza el coeficiente de silueta para determinar el número óptimo de clusters.
    Parameters:
        df (pd.DataFrame): Dataframe a utilizar.
        model_name (str): Nombre del modelo a utilizar.
        output_file_path (str): Path donde se guardará la visualización.
    Returns:
        optimal_k (int): Número óptimo de clusters.
    """
    k_range = range(2, 11)
    sil = []
    for k in k_range:
        gmm = GaussianMixture(n_components=k, random_state=42)
        labels = gmm.fit_predict(df)
        score = silhouette_score(df, labels)
        sil.append(score)

    optimal_k = k_range[0]
    max_sil = float('-inf')
    for i, k in enumerate(k_range):
        if 2 <= k <= 8 and sil[i] > max_sil:
            max_sil = sil[i]
            optimal_k = k

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sil, marker='o')
    plt.xlabel('NÚMERO DE CLUSTERS (k)')
    plt.ylabel('COEFICIENTE DE SILUETA')
    plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
    plt.title(f'MÉTODO DE COEF. SILUETA - {model_name}', fontsize=12)
    plt.legend()
    plt.savefig(output_file_path)
    plt.close()

    print('Silhouette score visualization saved')
    return optimal_k

def dbscan_silhouette(df, model_name, output_file_path):
    """
    Visualiza el coeficiente de silueta para determinar el número óptimo de clusters.
    Parameters:
        df (pd.DataFrame): Dataframe a utilizar.
        model_name (str): Nombre del modelo a utilizar.
        output_file_path (str): Path donde se guardará la visualización.
    Returns:
        optimal_k (int): Número óptimo de clusters.
    """
    k_range = range(2, 11)
    sil = []
    for k in k_range:
        dbs = DBSCAN(eps=k, min_samples=5)
        labels = dbs.fit(df)
        score = silhouette_score(df, labels)
        sil.append(score)

    optimal_k = k_range[0]
    max_sil = float('-inf')
    for i, k in enumerate(k_range):
        if 2 <= k <= 8 and sil[i] > max_sil:
            max_sil = sil[i]
            optimal_k = k

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, sil, marker='o')
    plt.xlabel('NÚMERO DE CLUSTERS (k)')
    plt.ylabel('COEFICIENTE DE SILUETA')
    plt.axvline(optimal_k, color='r', linestyle='--', label=f'Optimal k = {optimal_k}')
    plt.title(f'MÉTODO DE COEF. SILUETA - {model_name}', fontsize=12)
    plt.legend()
    plt.savefig(output_file_path)
    plt.close()

    print('Silhouette score visualization saved')
    return optimal_k


def silhoutte_method(df, model_name, output_file_path):
    '''
    Visualiza el coeficiente de silueta para determinar el número óptimo de clusters.
    Parameters:
        df (pd.DataFrame): Dataframe a utilizar.
        model_name (str): Nombre del modelo a utilizar.
        output_file_path (str): Path donde se guardará la visualización.
    Returns:
        optimal_k (int): Número óptimo de clusters.
    '''
    if not os.path.exists('../Segmentacion/Resultados'):
        os.makedirs('../Segmentacion/Resultados')
    
    if model_name == 'Kmeans':
        return K_means_silhouette(df, model_name, output_file_path)
    elif model_name == 'Agglomerative':
        return Agglomerative_silhouette(df, model_name, output_file_path)
    elif model_name == 'GaussianMixture':
        return gaussian_mixture_silhouette(df, model_name, output_file_path)
    elif model_name == 'DBSCAN':
        return dbscan_silhouette(df, model_name, output_file_path)



def get_clusters(df_original, df_scaled, model_name, k):
    '''
    Realiza el clustering de los datos.
    Parameters:
        df_original (pd.DataFrame): Dataframe original.
        df_scaled (pd.DataFrame): Dataframe a utilizar.
        k (int): Número de clusters.
        model_name (str): Nombre del modelo a utilizar.
    Returns:
        df (pd.DataFrame): Dataframe con la columna 'cluster' que indica el cluster al que pertenece cada fila.
    '''
    df_original_copy = df_original.copy()
    df_scaled_copy = df_scaled.copy()

    if model_name == 'Kmeans':
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_scaled_copy)
        df_original_copy['cluster'] = kmeans.labels_
        df_scaled_copy['cluster'] = kmeans.labels_
        return df_original_copy, df_scaled_copy
    elif model_name == 'Agglomerative':
        agglomerative = AgglomerativeClustering(n_clusters=k)
        labels = agglomerative.fit_predict(df_scaled_copy)
        df_original_copy['cluster'] = labels
        df_scaled_copy['cluster'] = labels
        return df_original_copy, df_scaled_copy
    elif model_name == 'GaussianMixture':
        gmm = GaussianMixture(n_components=k, random_state=42)
        labels = gmm.fit_predict(df_scaled_copy)
        df_original_copy['cluster'] = labels
        df_scaled_copy['cluster'] = labels
        return df_original_copy, df_scaled_copy
    elif model_name == 'DBSCAN':
        dbs = DBSCAN(eps=k, min_samples=5)
        labels = dbs.fit(df_scaled_copy)
        df_original_copy['cluster'] = labels
        df_scaled_copy['cluster'] = labels
        return df_original_copy, df_scaled_copy

def get_metrics(df_out, name):
    # Separar las características y los clusters
    X = df_out.drop(columns=['cluster'])  # Asegúrate de que solo estás eliminando la columna de cluster
    labels = df_out['cluster']

    # Calcular las métricas
    silhouette_avg = silhouette_score(X, labels)
    davies_bouldin_avg = davies_bouldin_score(X, labels)
    calinski_harabasz_avg = calinski_harabasz_score(X, labels)

    # Contar la cantidad de datos por cluster
    cluster_counts = labels.value_counts().sort_index()

    print(f'Metricas para {name}')
    print(f"Silhouette Score: {silhouette_avg}")
    # Int: Desde -1 (clustering pobre) a +1 (clustering perfecto)
    print(f"Davies-Bouldin Index: {davies_bouldin_avg}")
    # Int: Desde 0 (clustering perfecto) a infinito (Los números más bajos sugieren mejores soluciones de agrupación).
    print(f"Calinski-Harabasz Index: {calinski_harabasz_avg}")
    # Int: Desde 0 (clustering pobre) a infinito  (Los números más altos sugieren grupos mejor definidos).
    print("\nCantidad de datos por cluster:")
    for cluster, count in cluster_counts.items():
        print(f"Cluster {cluster}: {count} datos")


def plot_PCA(df_cluster_scaled, outpath_file_name):
    """
    Se encarga de graficar los clusters en 2D usando PCA. 
    
    Parameters:
        df_cluster_scaled (pd.DataFrame): Dataframe con las columnas 'cluster' escalado.
        outpath_name (str): Nombre del archivo de salida.
    
    Returns:
        None
    """
    df_copy = df_cluster_scaled.copy()

    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(df_copy.drop(['cluster'], axis=1))
    df_copy['pca-one'] = pca_result[:,0]
    df_copy['pca-two'] = pca_result[:,1]

    # graficar
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="pca-one", y="pca-two",
        hue="cluster",
        palette=sns.color_palette("hsv", df_copy['cluster'].nunique()),
        data=df_copy,
        legend="full",
        alpha=0.3
    )
    plt.title('PCA plot')

    # save file
    plt.savefig(outpath_file_name)

        
def plot_UMAP(df_cluster_scaled, outpath_file_name):
    """
    Se encarga de graficar los clusters en 2D usando UMAP.
    
    Parameters:
        df_cluster_scaled (pd.DataFrame): Dataframe con las columnas 'cluster' escalado.
        outpath_name (str): Nombre del archivo de salida.
    
    Returns:
        None
    """

    df_copy = df_cluster_scaled.copy()
    
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_result = reducer.fit_transform(df_copy.drop(['cluster'], axis=1))
    df_copy['umap-one'] = umap_result[:, 0]
    df_copy['umap-two'] = umap_result[:, 1]

    # graficar
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="umap-one", y="umap-two",
        hue="cluster",
        palette=sns.color_palette("hsv", df_copy['cluster'].nunique()),
        data=df_copy,
        legend="full",
        alpha=0.3
    )
    plt.title('UMAP plot')

    # save file
    plt.savefig(outpath_file_name)

