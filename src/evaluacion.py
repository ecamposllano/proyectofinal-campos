from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def evaluarModelo(rfm_escalado, etiquetas, nombre_modelo, excluir_ruido=False):

    if excluir_ruido:
        mascara = etiquetas != -1
        datos = rfm_escalado[mascara]
        etiquetas_filtradas = etiquetas[mascara]
    else:
        datos = rfm_escalado
        etiquetas_filtradas = etiquetas

    silhouette = silhouette_score(datos, etiquetas_filtradas)
    davies_bouldin = davies_bouldin_score(datos, etiquetas_filtradas)
    calinski_harabasz = calinski_harabasz_score(datos, etiquetas_filtradas)

    print(f"Modelo: {nombre_modelo}")
    print(f"  Silhouette: {silhouette:.4f}")
    print(f"  Davies-Bouldin: {davies_bouldin:.4f}")
    print(f"  Calinski-Harabasz: {calinski_harabasz:.4f}")

    return {
        "modelo": nombre_modelo,
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "calinski_harabasz": calinski_harabasz
    }