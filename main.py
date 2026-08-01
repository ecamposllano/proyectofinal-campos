import numpy as np

from src.cargar_datos import cargarDatos
from src.analisis_inicial import resumenInicial
from src.limpieza import limpiarDatos,resumenLimpieza
from src.analisis_paises import analisisPaises, compararPaises
from src.graficar import graficarBarrasPaises, graficarBarrasPaisesSinUK


def main():
    # 1. CARGA DE DATOS
    df = cargarDatos("datos/online_retail_II.xlsx")

    # 2. AUDITORIA DE CALIDAD
    resumenInicial(df)

    # 3. LIMPIEZA DEL DATASET
    df_limpio = limpiarDatos(df)
    resumenLimpieza(df, df_limpio)

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

    df_uk = df_limpio.copy()
    df_uk["Country"] = np.where(df_uk["Country"] == "United Kingdom", "United Kingdom", "Resto del mundo")
    compararPaises(df_uk, "United Kingdom", "Resto del mundo")

if __name__ == "__main__":
    main()