import pandas as pd
from sklearn.cluster import KMeans

def buscarKOptimo(datos, k_min=2, k_max=10):
    resultados = []

    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(datos)

        resultados.append({"k": k, "inercia": kmeans.inertia_})
 
    return pd.DataFrame(resultados)

def aplicarKMeans(rfm, rfm_escalado, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiquetas = kmeans.fit_predict(rfm_escalado)

    rfm_con_cluster = rfm.copy()
    rfm_con_cluster["Cluster_KMeans"] = etiquetas

    print(f"K-means con k={k}")
    print(rfm_con_cluster["Cluster_KMeans"].value_counts().sort_index())

    return rfm_con_cluster, kmeans


def resumenKMeans(rfm_k):
    resumen = rfm_k.groupby("Cluster_KMeans").agg(
        cantidad_clientes=("Customer ID", "count"),
        recencia_promedio=("Recencia", "mean"),
        frecuencia_promedio=("Frecuencia", "mean"),
        monto_promedio=("Monto", "mean")
    ).reset_index()

    print(resumen)