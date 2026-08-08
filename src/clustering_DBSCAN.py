import pandas as pd
from sklearn.cluster import DBSCAN


def buscarParametrosDBSCAN(rfm_escalado, eps_valores, min_puntos_valores):
    resultados = []

    for eps in eps_valores:
        for min_puntos in min_puntos_valores:
            dbscan = DBSCAN(eps=eps, min_samples=min_puntos)
            etiquetas = dbscan.fit_predict(rfm_escalado)

            valores_unicos = set(etiquetas)
            hay_ruido = -1 in valores_unicos

            n_clusters = len(valores_unicos)

            if hay_ruido:
                n_clusters = n_clusters - 1

            n_ruido = (etiquetas == -1).sum()
            porcentaje_ruido = round(100 * n_ruido / len(etiquetas), 2)

            resultados.append({
                "eps": eps,
                "min_puntos": min_puntos,
                "n_clusters": n_clusters,
                "n_ruido": n_ruido,
                "porcentaje_ruido": porcentaje_ruido
            })

    return pd.DataFrame(resultados)


def aplicarDBSCAN(rfm, rfm_escalado, eps, min_puntos):
    dbscan = DBSCAN(eps=eps, min_samples=min_puntos)
    etiquetas = dbscan.fit_predict(rfm_escalado)

    rfm_con_cluster = rfm.copy()
    rfm_con_cluster["Cluster_DBSCAN"] = etiquetas

    n_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
    n_ruido = (etiquetas == -1).sum()

    print(f"DBSCAN con eps={eps}, min_puntos={min_puntos}")
    print(f"Clusters encontrados: {n_clusters}")
    print(f"Puntos de ruido: {n_ruido}")
    print(rfm_con_cluster["Cluster_DBSCAN"].value_counts().sort_index())

    return rfm_con_cluster, dbscan