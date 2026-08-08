import pandas as pd
import hdbscan


def buscarParametrosHDBSCAN(rfm_escalado, min_cluster_size_valores, min_puntos_valores):
    resultados = []

    for min_cluster_size in min_cluster_size_valores:
        for min_puntos in min_puntos_valores:
            clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_puntos)
            etiquetas = clusterer.fit_predict(rfm_escalado)

            valores_unicos = set(etiquetas)
            hay_ruido = -1 in valores_unicos

            n_clusters = len(valores_unicos)
            if hay_ruido:
                n_clusters = n_clusters - 1

            n_ruido = (etiquetas == -1).sum()
            porcentaje_ruido = round(100 * n_ruido / len(etiquetas), 2)

            resultados.append({
                "min_cluster_size": min_cluster_size,
                "min_samples": min_puntos,
                "n_clusters": n_clusters,
                "n_ruido": n_ruido,
                "porcentaje_ruido": porcentaje_ruido
            })

    return pd.DataFrame(resultados)


def aplicarHDBSCAN(rfm, rfm_escalado, min_cluster_size, min_puntos):
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_puntos)
    etiquetas = clusterer.fit_predict(rfm_escalado)

    rfm_con_cluster = rfm.copy()
    rfm_con_cluster["Cluster_HDBSCAN"] = etiquetas

    valores_unicos = set(etiquetas)
    hay_ruido = -1 in valores_unicos

    n_clusters = len(valores_unicos)
    if hay_ruido:
        n_clusters = n_clusters - 1

    n_ruido = (etiquetas == -1).sum()

    print(f"HDBSCAN ajustado con min_cluster_size={min_cluster_size}, min_samples={min_puntos}")
    print(f"Clusters encontrados: {n_clusters}")
    print(f"Puntos de ruido: {n_ruido}")
    print(rfm_con_cluster["Cluster_HDBSCAN"].value_counts().sort_index())

    return rfm_con_cluster, clusterer