import numpy as np
import pandas as pd

from src.cargar_datos import cargarDatos
from src.analisis_inicial import resumenInicial
from src.limpieza import limpiarDatos,resumenLimpieza
from src.analisis_paises import analisisPaises, compararPaises
from src.graficar import graficarBarrasPaises, graficarBarrasPaisesSinUK, graficarCodo, graficarArbolExpansionMinima 
from src.graficar import graficarJerarquiaClusters, graficarRFM3D, graficarBoxplotRFMEscalado, graficarClustersRFM2D
from src.graficar import graficarClustersRFM2DZoom
from src.rfm import calcularRFM, resumenRFM
from src.escalamiento import escalarDatos, resumenEscalamiento
from src.clustering_KMeans import buscarKOptimo, aplicarKMeans, resumenKMeans
from src.clustering_DBSCAN import buscarParametrosDBSCAN, aplicarDBSCAN
from src.clustering_HDBSCAN import buscarParametrosHDBSCAN, aplicarHDBSCAN
from src.evaluacion import evaluarModelo


def main():
    # CARGA DE DATOS
    df = cargarDatos("datos/online_retail_II.xlsx")

    # AUDITORIA DE CALIDAD
    resumenInicial(df)

    # LIMPIEZA DEL DATASET
    df_limpio = limpiarDatos(df)
    resumenLimpieza(df, df_limpio)
    """
    # DISTRIBUCION DE PAISES EN EL DATASET
    resumenPaises = analisisPaises(df_limpio)
    graficarBarrasPaises(resumenPaises)
    graficarBarrasPaisesSinUK(resumenPaises)


    # COMPARACION DE PAISES
    compararPaises(df_limpio, "United Kingdom", "Lithuania")
    compararPaises(df_limpio, "France", "Germany")
    compararPaises(df_limpio, "United Arab Emirates", "Portugal")
    compararPaises(df_limpio, "Netherlands", "Germany")
    compararPaises(df_limpio, "Japan", "Switzerland")
    """
    df_uk = df_limpio.copy()
    df_uk["Country"] = np.where(df_uk["Country"] == "United Kingdom", "United Kingdom", "Resto del mundo")
    compararPaises(df_uk, "United Kingdom", "Resto del mundo")

    df_uk = df_uk[df_uk["Country"] == "United Kingdom"]

    # MODELO RFM
    rfm = calcularRFM(df_uk)
    resumenRFM(rfm)

    # ESCALAMOS LOS DATOS
    rfm_escalado, scaler = escalarDatos(rfm)
    resumenEscalamiento(rfm_escalado)
    graficarBoxplotRFMEscalado(rfm_escalado)


    # CLUSTERING - KMEANS
    resultados_k = buscarKOptimo(rfm_escalado)
    #graficarCodo(resultados_k)
    
    rfm_kmeans4, modelo_kmeans4 = aplicarKMeans(rfm, rfm_escalado, k=4)
    """
    resumenKMeans(rfm_kmeans4)
    graficarRFM3D(rfm_kmeans4, rfm_kmeans4["Cluster_KMeans"], "Clusters K-means (k=4)", "rfm_3d_kmeans_k4.png")
    graficarClustersRFM2D(rfm_kmeans4, 'Recencia', 'Frecuencia', 'Cluster_KMeans', 'clusters_recencia_frecuencia_k4.png')
    graficarClustersRFM2D(rfm_kmeans4, 'Recencia', 'Monto', 'Cluster_KMeans', 'clusters_recencia_monto_k4.png')
    graficarClustersRFM2D(rfm_kmeans4, 'Frecuencia', 'Monto', 'Cluster_KMeans', 'clusters_frecuencia_monto_k4.png')
    graficarClustersRFM2DZoom(rfm_kmeans4, 'Frecuencia', 'Monto', 'Cluster_KMeans', 'clusters_frecuencia_monto_zoom.png')
    graficarClustersRFM2DZoom(rfm_kmeans4, 'Recencia', 'Monto', 'Cluster_KMeans', 'clusters_recencia_monto_zoom.png')
    """
    rfm_kmeans5, modelo_kmeans5 = aplicarKMeans(rfm, rfm_escalado, k=5)
    """
    resumenKMeans(rfm_kmeans5)
    graficarRFM3D(rfm_kmeans5, rfm_kmeans5["Cluster_KMeans"], "Clusters K-means (k=5)", "rfm_3d_kmeans_k5.png")
    graficarClustersRFM2D(rfm_kmeans5, 'Recencia', 'Frecuencia', 'Cluster_KMeans', 'clusters_recencia_frecuencia_k5.png')
    graficarClustersRFM2D(rfm_kmeans5, 'Recencia', 'Monto', 'Cluster_KMeans', 'clusters_recencia_monto_k5.png')
    graficarClustersRFM2D(rfm_kmeans5, 'Frecuencia', 'Monto', 'Cluster_KMeans', 'clusters_frecuencia_monto_k5.png')
    graficarClustersRFM2DZoom(rfm_kmeans5, 'Frecuencia', 'Monto', 'Cluster_KMeans', 'clusters_frecuencia_monto_zoom_k5.png')
    graficarClustersRFM2DZoom(rfm_kmeans5, 'Recencia', 'Monto', 'Cluster_KMeans', 'clusters_recencia_monto_zoom_k5.png')
    """
    # CLUSTERING - DBSCAN
    resultados_dbscan = buscarParametrosDBSCAN(
        rfm_escalado,
        eps_valores=[0.2, 0.3, 0.4, 0.7, 1, 1.5],
        min_puntos_valores=[3, 5, 10, 15]
    )
    print(resultados_dbscan)

    rfm_dbscan, modelo_dbscan = aplicarDBSCAN(rfm, rfm_escalado, eps=0.2, min_puntos=5)

    resumen_dbscan = rfm_dbscan.groupby("Cluster_DBSCAN").agg(
        cantidad_clientes=("Customer ID", "count"),
        recencia_promedio=("Recencia", "mean"),
        frecuencia_promedio=("Frecuencia", "mean"),
        monto_promedio=("Monto", "mean")
    ).reset_index()
    """
    print(resumen_dbscan)
    graficarRFM3D(rfm_dbscan, rfm_dbscan["Cluster_DBSCAN"], "Clusters DBSCAN", "rfm_3d_dbscan.png")
    graficarClustersRFM2D(rfm_dbscan, 'Recencia', 'Frecuencia', "Cluster_DBSCAN", 'clusters_recencia_frecuencia_dbscan.png')
    graficarClustersRFM2D(rfm_dbscan, 'Recencia', 'Monto', "Cluster_DBSCAN", 'clusters_recencia_monto_dbscan.png')
    graficarClustersRFM2D(rfm_dbscan, 'Frecuencia', 'Monto', "Cluster_DBSCAN", 'clusters_frecuencia_monto_dbscan.png')
    """
    # CLUSTERING - HDBSCAN
    resultados_hdbscan = buscarParametrosHDBSCAN(
        rfm_escalado,
        min_cluster_size_valores=[100, 150, 200, 300, 500],
        min_puntos_valores=[1, 2, 3, 5, 10]
    )
    print(resultados_hdbscan)

    rfm_hdbscan, modelo_hdbscan = aplicarHDBSCAN(rfm, rfm_escalado, min_cluster_size=500, min_puntos=3)
    """
    graficarRFM3D(rfm_hdbscan, rfm_hdbscan["Cluster_HDBSCAN"], "Clusters HDBSCAN", "rfm_3d_hdbscan.png")
    graficarClustersRFM2D(rfm_hdbscan, 'Recencia', 'Frecuencia', "Cluster_HDBSCAN", 'clusters_recencia_frecuencia_hdbscan.png')
    graficarClustersRFM2D(rfm_hdbscan, 'Recencia', 'Monto', "Cluster_HDBSCAN", 'clusters_recencia_monto_hdbscan.png')
    graficarClustersRFM2D(rfm_hdbscan, 'Frecuencia', 'Monto', "Cluster_HDBSCAN", 'clusters_frecuencia_monto_hdbscan.png')
    graficarClustersRFM2DZoom(rfm_hdbscan, 'Frecuencia', 'Monto', 'Cluster_HDBSCAN', 'clusters_frecuencia_monto_zoom_hdbscan.png')
    graficarClustersRFM2DZoom(rfm_hdbscan, 'Recencia', 'Monto', 'Cluster_HDBSCAN', 'clusters_recencia_monto_zoom_hdbscan.png')
    
    graficarJerarquiaClusters(modelo_hdbscan)
    """
    resumen_hdbscan = rfm_hdbscan.groupby("Cluster_HDBSCAN").agg(
        cantidad_clientes=("Customer ID", "count"),
        recencia_promedio=("Recencia", "mean"),
        frecuencia_promedio=("Frecuencia", "mean"),
        monto_promedio=("Monto", "mean")
    ).reset_index()

    print(resumen_hdbscan)

    rfm_hdbscan2, modelo_hdbscan2 = aplicarHDBSCAN(rfm, rfm_escalado, min_cluster_size=200, min_puntos=1)
    """
    graficarRFM3D(rfm_hdbscan2, rfm_hdbscan2["Cluster_HDBSCAN"], "Clusters HDBSCAN", "rfm_3d_hdbscan2.png")
    graficarClustersRFM2D(rfm_hdbscan2, 'Recencia', 'Frecuencia', "Cluster_HDBSCAN", 'clusters_recencia_frecuencia_hdbscan2.png')
    graficarClustersRFM2D(rfm_hdbscan2, 'Recencia', 'Monto', "Cluster_HDBSCAN", 'clusters_recencia_monto_hdbscan2.png')
    graficarClustersRFM2D(rfm_hdbscan2, 'Frecuencia', 'Monto', "Cluster_HDBSCAN", 'clusters_frecuencia_monto_hdbscan2.png')
    graficarClustersRFM2DZoom(rfm_hdbscan2, 'Frecuencia', 'Monto', 'Cluster_HDBSCAN', 'clusters_frecuencia_monto_zoom_hdbscan2.png')
    graficarClustersRFM2DZoom(rfm_hdbscan2, 'Recencia', 'Monto', 'Cluster_HDBSCAN', 'clusters_recencia_monto_zoom_hdbscan2.png')
        
    graficarJerarquiaClusters(modelo_hdbscan2)
    """
    resumen_hdbscan2 = rfm_hdbscan2.groupby("Cluster_HDBSCAN").agg(
        cantidad_clientes=("Customer ID", "count"),
        recencia_promedio=("Recencia", "mean"),
        frecuencia_promedio=("Frecuencia", "mean"),
        monto_promedio=("Monto", "mean")
    ).reset_index()

    print(resumen_hdbscan2)

    # EVALUACION DE LOS MODELOS
    resultados_evaluacion = []

    resultados_evaluacion.append(evaluarModelo(rfm_escalado, rfm_kmeans4['Cluster_KMeans'], "K-Means (k=4)"))
    resultados_evaluacion.append(evaluarModelo(rfm_escalado, rfm_kmeans5['Cluster_KMeans'], "K-Means (k=5)"))
    resultados_evaluacion.append(evaluarModelo(rfm_escalado, rfm_dbscan['Cluster_DBSCAN'], "DBSCAN", excluir_ruido=True))
    resultados_evaluacion.append(evaluarModelo(rfm_escalado, rfm_hdbscan['Cluster_HDBSCAN'], "HDBSCAN (500/3)", excluir_ruido=True))
    resultados_evaluacion.append(evaluarModelo(rfm_escalado, rfm_hdbscan2['Cluster_HDBSCAN'], "HDBSCAN (200/1)", excluir_ruido=True))

    df_evaluacion = pd.DataFrame(resultados_evaluacion)
    print(df_evaluacion)


if __name__ == "__main__":
    main()