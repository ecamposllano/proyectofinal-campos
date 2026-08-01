import os
import pandas as pd


def cargarDatos(ruta, ruta_cache="datos/online_retail_II_cache.csv"):

    if os.path.exists(ruta_cache):
        print(f"Leyendo desde cache: {ruta_cache}")
        df = pd.read_csv(ruta_cache, parse_dates=["InvoiceDate"])
        return df

    print("No hay cache todavía, leyendo el Excel original")
    hoja_1 = pd.read_excel(ruta, sheet_name="Year 2009-2010")
    hoja_2 = pd.read_excel(ruta, sheet_name="Year 2010-2011")

    df = pd.concat([hoja_1, hoja_2], ignore_index=True)

    print(f"Guardando cache en: {ruta_cache}")
    df.to_csv(ruta_cache, index=False)

    return df