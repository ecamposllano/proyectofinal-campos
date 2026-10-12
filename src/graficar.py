import os
import matplotlib.pyplot as plt
import numpy as np


def graficarBarrasPaises(resumenPaises, top_n=15):

    os.makedirs("graficos", exist_ok=True)

    datos_ordenados = resumenPaises["filas"].sort_values(ascending=True).tail(top_n)

    plt.figure(figsize=(9, 7))
    plt.barh(datos_ordenados.index, datos_ordenados.values)
    plt.xlabel("Cantidad de registros")
    plt.tight_layout()

    ruta = "graficos/barras_paises.png"
    plt.savefig(ruta, dpi=150)
    plt.show()


def graficarBarrasPaisesSinUK(resumenPaises, top_n=15):

    os.makedirs("graficos", exist_ok=True)

    datos_sin_uk = resumenPaises["filas"].drop("Reino Unido", errors="ignore")
    datos_ordenados = datos_sin_uk.sort_values(ascending=True).tail(top_n)

    plt.figure(figsize=(9, 7))
    plt.barh(datos_ordenados.index, datos_ordenados.values)
    plt.xlabel("Cantidad de registros")
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
    plt.savefig("graficos/arbol_expansion_minima.png", dpi=150, bbox_inches="tight")
    plt.show()


def graficarJerarquiaClusters(clusterer):
    os.makedirs("graficos", exist_ok=True)
    plt.figure(figsize=(12, 8))
    clusterer.condensed_tree_.plot()
    plt.savefig("graficos/jerarquia_clusters.png", dpi=150, bbox_inches="tight")
    plt.show()


def graficarRFM3D(rfm, etiquetas=None, titulo="Distribución de clientes según RFM", nombre_archivo="rfm_3d.png"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colores = ["#1F4E9E", "#F2B600", "#2E9E44", "#D7263D", "#00B8D4", "#7B4A12", "#E91E8C", "#000000"]

    if etiquetas is None:
        ax.scatter(
            rfm["Recencia"], rfm["Frecuencia"], rfm["Monto"],
            color=colores[0], s=10, alpha=0.3
        )
    else:
        for cluster in sorted(etiquetas.unique()):
            mask = etiquetas == cluster
            if cluster == -1:
                color, nombre = "darkgray", "Cluster -1"
            else:
                color, nombre = colores[cluster % len(colores)], f"Cluster {cluster}"
            ax.scatter(
                rfm.loc[mask, "Recencia"],
                rfm.loc[mask, "Frecuencia"],
                rfm.loc[mask, "Monto"],
                color=color, s=10, alpha=0.6, label=nombre
            )
        n = len(etiquetas.unique())
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels, title="Cluster", loc="lower center",
            ncol=min(n, 5), markerscale=3, frameon=True)
        fig.subplots_adjust(bottom=0.2)

    ax.set_xlabel("Recencia (días)", labelpad=10)
    ax.set_ylabel("Frecuencia", labelpad=10)
    ax.set_zlabel("Monto", labelpad=15)
    #ax.set_title(titulo)
    ax.tick_params(axis="z", pad=8, labelsize=8)
    ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x/1000:.0f}"))
    
    plt.savefig(f"graficos/{nombre_archivo}", dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.show()


def graficarBoxplotRFMEscalado(rfm_escalado):
    columnas_rfm = ['Recencia', 'Frecuencia', 'Monto']
    colores = plt.cm.tab10.colors[:3]

    versiones = [
        ("graficos/boxplot_rfm_escalado.png", None),
        ("graficos/boxplot_rfm_escalado_zoom.png", (-1, 10)),
    ]

    for ruta, limite_y in versiones:
        plt.figure(figsize=(8, 6))
        cajas = rfm_escalado[columnas_rfm].boxplot(
            patch_artist=True,
            flierprops=dict(marker='o', markersize=2, alpha=0.5),
            medianprops=dict(color='black'),
            return_type='dict'
        )

        for caja, color in zip(cajas['boxes'], colores):
            caja.set_facecolor(color)
            caja.set_alpha(0.7)

        if limite_y is not None:
            plt.ylim(limite_y)

        plt.ylabel('Valor escalado (z-score)')
        plt.grid(True, alpha=0.3)
        plt.savefig(ruta, dpi=300)
        plt.show()

COLORES_CLUSTERS = ["#1F4E9E", "#F2B600", "#2E9E44", "#D7263D", "#00B8D4", "#7B4A12", "#E91E8C", "#000000"]
COLOR_RUIDO = "darkgray"

def graficarClustersRFM2D(rfm_con_cluster, columna_x, columna_y, columna_cluster, nombre_archivo):
    plt.figure(figsize=(8, 6))

    clusters_unicos = sorted(rfm_con_cluster[columna_cluster].unique())

    for cluster in clusters_unicos:
        subset = rfm_con_cluster[rfm_con_cluster[columna_cluster] == cluster]
        if cluster == -1:
            color, nombre = COLOR_RUIDO, "Cluster -1"
        else:
            color, nombre = COLORES_CLUSTERS[cluster % len(COLORES_CLUSTERS)], f"Cluster {cluster}"
        plt.scatter(
            subset[columna_x],
            subset[columna_y],
            color=color,
            alpha=0.6,
            edgecolors='k',
            linewidths=0.3,
            label=nombre
        )

    plt.xlabel(columna_x)
    plt.ylabel(columna_y)
    n = len(clusters_unicos)
    plt.legend(
        title='Cluster', loc='upper left', bbox_to_anchor=(1.02, 1),
        ncol=1 if n <= 10 else 2)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"graficos/{nombre_archivo}", bbox_inches="tight")
    plt.show()


def graficarClustersRFM2DZoom(rfm_con_cluster, columna_x, columna_y, columna_cluster, nombre_archivo, percentil=99):
    """
    Genera un scatter plot 2D entre dos variables RFM, coloreado por cluster,
    con zoom para dejar fuera de la vista los valores mas extremos y poder
    apreciar mejor la separacion entre los clusters.
    """
    plt.figure(figsize=(8, 6))

    clusters_unicos = sorted(rfm_con_cluster[columna_cluster].unique())

    for cluster in clusters_unicos:
        subset = rfm_con_cluster[rfm_con_cluster[columna_cluster] == cluster]
        if cluster == -1:
            color, nombre = COLOR_RUIDO, "Cluster -1"
        else:
            color, nombre = COLORES_CLUSTERS[cluster % len(COLORES_CLUSTERS)], f"Cluster {cluster}"
        plt.scatter(
            subset[columna_x],
            subset[columna_y],
            color=color,
            alpha=0.6,
            edgecolors='k',
            linewidths=0.3,
            label=nombre
        )

    limite_x = np.percentile(rfm_con_cluster[columna_x], percentil)
    limite_y = np.percentile(rfm_con_cluster[columna_y], percentil)
    plt.xlim(0, limite_x)
    plt.ylim(0, limite_y)

    plt.xlabel(columna_x)
    plt.ylabel(columna_y)
    n = len(clusters_unicos)
    plt.legend(
        title='Cluster', loc='upper left', bbox_to_anchor=(1.02, 1),
        ncol=1 if n <= 10 else 2)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"graficos/{nombre_archivo}", bbox_inches="tight")
    plt.show()