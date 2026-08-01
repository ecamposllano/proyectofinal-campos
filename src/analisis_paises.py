import pandas as pd
import numpy as np

def analisisPaises(df):

    # Conteo y porcentaje de filas por país
    conteo_paises = df["Country"].value_counts()
    porcentaje_paises = (conteo_paises / len(df) * 100).round(2)

    # Facturación por país
    df_paises = df.copy()
    df_paises["ImporteTotal"] = df_paises["Quantity"] * df_paises["Price"]
    facturacion_por_pais = df_paises.groupby("Country")["ImporteTotal"].sum()
    porcentaje_facturacion = (facturacion_por_pais / facturacion_por_pais.sum() * 100).round(2)

    print("\n--- Resumen por país ---")
    resumen_paises = pd.DataFrame({
        "Importe Total": facturacion_por_pais,
        "porcentaje Importe": porcentaje_facturacion,
        "filas": conteo_paises,
        "porcentaje Filas": porcentaje_paises
    })
    resumen_paises = resumen_paises.sort_values("Importe Total", ascending=False)
    print(resumen_paises)

    return resumen_paises

def compararPaises(df, pais_1, pais_2):

    df_paises = df.copy()
    df_paises["ImporteTotal"] = df_paises["Quantity"] * df_paises["Price"]

    resultados = {}

    for pais in (pais_1, pais_2):
        datos_pais = df_paises[df_paises["Country"] == pais]

        cant_filas = len(datos_pais)
        cant_facturas = datos_pais["Invoice"].nunique()
        cant_clientes = datos_pais["Customer ID"].nunique()

        importe_promedio = datos_pais["ImporteTotal"].mean()
        cantidad_promedio = datos_pais["Quantity"].mean()
        precio_promedio = datos_pais["Price"].mean()

        importe_por_factura = datos_pais.groupby("Invoice")["ImporteTotal"].sum()
        importe_por_factura_promedio = importe_por_factura.mean()

        items_por_factura = datos_pais.groupby("Invoice")["Quantity"].sum()
        items_promedio_factura = items_por_factura.mean()

        resultados[pais] = {
            "filas": cant_filas,
            "facturas unicas": cant_facturas,
            "clientes unicos": cant_clientes,
            "importe promedio": round(importe_promedio, 2),
            "cantidad promedio": round(cantidad_promedio, 2),
            "precio promedio": round(precio_promedio, 2),
            "imporet promedio por factura": round(importe_por_factura_promedio, 2),
            "items promedio por factura": round(items_promedio_factura, 2),
        }

    comparacion = pd.DataFrame(resultados)
    print(f"\n--- Comparación: {pais_1} vs {pais_2} ---")
    print(comparacion)

    return comparacion
