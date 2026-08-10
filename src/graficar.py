import os
import matplotlib.pyplot as plt


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