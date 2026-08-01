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