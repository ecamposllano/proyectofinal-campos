import numpy as np

from src.cargar_datos import cargarDatos
from src.analisis_inicial import resumenInicial
from src.limpieza import limpiarDatos,resumenLimpieza
from src.analisis_paises import analisisPaises, compararPaises
from src.graficar import graficarBarrasPaises, graficarBarrasPaisesSinUK, graficarCodo
from src.rfm import calcularRFM, resumenRFM
from src.escalamiento import escalarDatos, resumenEscalamiento
from src.clustering_KMeans import buscarKOptimo, aplicarKMeans
from src.clustering_DBSCAN import buscarParametrosDBSCAN, aplicarDBSCAN
from src.clustering_HDBSCAN import buscarParametrosHDBSCAN, aplicarHDBSCAN


def main():
    # CARGA DE DATOS
    df = cargarDatos("datos/online_retail_II.xlsx")
    """
    # AUDITORIA DE CALIDAD
    resumenInicial(df)
    """
    # LIMPIEZA DEL DATASET
    df_limpio = limpiarDatos(df)
#    resumenLimpieza(df, df_limpio)
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
    #compararPaises(df_uk, "United Kingdom", "Resto del mundo")

    df_uk = df_uk[df_uk["Country"] == "United Kingdom"]

    # MODELO RFM
    rfm = calcularRFM(df_uk)
    resumenRFM(rfm)

    # ESCALAMOS LOS DATOS
    rfm_escalado, scaler = escalarDatos(rfm)
    resumenEscalamiento(rfm_escalado)

    # CLUSTERING - KMEANS
    resultados_k = buscarKOptimo(rfm_escalado)
    graficarCodo(resultados_k)

    rfm_kmeans4, modelo_kmeans4 = aplicarKMeans(rfm, rfm_escalado, k=4)
    rfm_kmeans5, modelo_kmeans5 = aplicarKMeans(rfm, rfm_escalado, k=5)

    # CLUSTERING - DBSCAN
    resultados_dbscan = buscarParametrosDBSCAN(
        rfm_escalado,
        eps_valores=[0.2, 0.3, 0.4, 0.7, 1, 1.5],
        min_puntos_valores=[3, 5, 10, 15]
    )
    print(resultados_dbscan)

    rfm_dbscan, modelo_dbscan = aplicarDBSCAN(rfm, rfm_escalado, eps=1.5, min_puntos=3)

    # CLUSTERING - HDBSCAN
    resultados_hdbscan = buscarParametrosHDBSCAN(
        rfm_escalado,
        min_cluster_size_valores=[100, 150, 200, 300, 500],
        min_puntos_valores=[1, 2, 3, 5, 10]
    )
    print(resultados_hdbscan)

if __name__ == "__main__":
    main()