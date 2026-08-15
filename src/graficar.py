import os
import matplotlib.pyplot as plt
import numpy as np


def graficarBarrasPaises(resumenPaises, top_n=15):

    os.makedirs("graficos", exist_ok=True)

    datos_ordenados = resumenPaises["filas"].sort_values(ascending=True).tail(top_n)

    plt.figure(figsize=(9, 7))
    plt.barh(datos_ordenados.index, datos_ordenados.values)
    plt.xlabel("Cantidad de filas")
    plt.title(f"Top {top_n} países por cantidad de transacciones")
    plt.tight_layout()

    ruta = "graficos/barras_paises.png"
    plt.savefig(ruta, dpi=150)
    plt.show()


def graficarBarrasPaisesSinUK(resumenPaises, top_n=15):

    os.makedirs("graficos", exist_ok=True)

    datos_sin_uk = resumenPaises["filas"].drop("United Kingdom", errors="ignore")
    datos_ordenados = datos_sin_uk.sort_values(ascending=True).tail(top_n)

    plt.figure(figsize=(9, 7))
    plt.barh(datos_ordenados.index, datos_ordenados.values)
    plt.xlabel("Cantidad de filas")
    plt.title(f"Top {top_n} países por cantidad de transacciones (sin UK)")
    plt.tight_layout()

    ruta = "graficos/barras_paises_sin_uk.png"
    plt.savefig(ruta, dpi=150)
    plt.show()


def graficarCodo(resultados_k):
    os.makedirs("graficos", exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(resultados_k["k"], resultados_k["inercia"], marker="o")
    plt.xlabel("Cantidad de clusters (k)")
    plt.ylabel("Inercia")
    plt.title("Método del codo")
    plt.xticks(resultados_k["k"])
    plt.grid(True)
    plt.savefig("graficos/codo_k.png", dpi=150)
    plt.show()

 
def graficarArbolExpansionMinima(clusterer):
    os.makedirs("graficos", exist_ok=True)
    plt.figure(figsize=(10, 8))
    clusterer.minimum_spanning_tree_.plot(
        edge_cmap="viridis",
        edge_alpha=0.6,
        node_size=5,
        edge_linewidth=1
    )
    plt.title("Árbol de expansión mínima (HDBSCAN)")
    plt.savefig("graficos/arbol_expansion_minima.png", dpi=150, bbox_inches="tight")
    plt.show()


def graficarJerarquiaClusters(clusterer):
    os.makedirs("graficos", exist_ok=True)
    plt.figure(figsize=(12, 8))
    clusterer.condensed_tree_.plot()
    plt.title("Jerarquía de clusters (HDBSCAN)")
    plt.savefig("graficos/jerarquia_clusters.png", dpi=150, bbox_inches="tight")
    plt.show()


def graficarRFM3D(rfm, etiquetas=None, titulo="Distribución de clientes según RFM", nombre_archivo="rfm_3d.png"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    clusters_unicos = sorted(etiquetas.unique())
    colores = plt.cm.tab10.colors

    for i, cluster in enumerate(clusters_unicos):
        mask = etiquetas == cluster
        ax.scatter(
            rfm.loc[mask, "Recencia"],
            rfm.loc[mask, "Frecuencia"],
            rfm.loc[mask, "Monto"],
            color=colores[i],
            s=10,
            alpha=0.6,
            label=f'Cluster {cluster}'
        )

    ax.set_xlabel("Recencia (días)")
    ax.set_ylabel("Frecuencia")
    ax.set_zlabel("Monto")
    ax.set_title(titulo)
    ax.legend(title='Cluster')
    plt.savefig(f"graficos/{nombre_archivo}")
    plt.show()


def graficarBoxplotRFMEscalado(rfm_escalado):
    columnas_rfm = ['Recencia', 'Frecuencia', 'Monto']
    
    plt.figure(figsize=(8, 6))
    rfm_escalado[columnas_rfm].boxplot()
    plt.title('Distribucion de variables RFM escaladas')
    plt.ylabel('Valor escalado (z-score)')
    plt.xlabel('Variable')
    plt.grid(True, alpha=0.3)
    plt.savefig("graficos/boxplot_rfm_escalado.png")
    plt.show()


def graficarClustersRFM2D(rfm_con_cluster, columna_x, columna_y, columna_cluster, nombre_archivo):
    plt.figure(figsize=(8, 6))
    
    clusters_unicos = sorted(rfm_con_cluster[columna_cluster].unique())
    colores = plt.cm.tab10.colors

    for i, cluster in enumerate(clusters_unicos):
        subset = rfm_con_cluster[rfm_con_cluster[columna_cluster] == cluster]
        plt.scatter(
            subset[columna_x],
            subset[columna_y],
            color=colores[i],
            alpha=0.6,
            edgecolors='k',
            linewidths=0.3,
            label=f'Cluster {cluster}'
        )

    plt.title(f'Clusters segun {columna_x} y {columna_y}')
    plt.xlabel(columna_x)
    plt.ylabel(columna_y)
    plt.legend(title='Cluster')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"graficos/{nombre_archivo}")
    plt.show()


def graficarClustersRFM2DZoom(rfm_con_cluster, columna_x, columna_y, columna_cluster, nombre_archivo, percentil=99):
    """
    Genera un scatter plot 2D entre dos variables RFM, coloreado por cluster,
    con zoom para dejar fuera de la vista los valores mas extremos y poder
    apreciar mejor la separacion entre los clusters.
    """
    plt.figure(figsize=(8, 6))
    
    clusters_unicos = sorted(rfm_con_cluster[columna_cluster].unique())
    colores = plt.cm.tab10.colors

    for i, cluster in enumerate(clusters_unicos):
        subset = rfm_con_cluster[rfm_con_cluster[columna_cluster] == cluster]
        plt.scatter(
            subset[columna_x],
            subset[columna_y],
            color=colores[i],
            alpha=0.6,
            edgecolors='k',
            linewidths=0.3,
            label=f'Cluster {cluster}'
        )

    limite_x = np.percentile(rfm_con_cluster[columna_x], percentil)
    limite_y = np.percentile(rfm_con_cluster[columna_y], percentil)
    plt.xlim(0, limite_x)
    plt.ylim(0, limite_y)

    plt.title(f'Clusters segun {columna_x} y {columna_y} (zoom)')
    plt.xlabel(columna_x)
    plt.ylabel(columna_y)
    plt.legend(title='Cluster')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"graficos/{nombre_archivo}")
    plt.show()