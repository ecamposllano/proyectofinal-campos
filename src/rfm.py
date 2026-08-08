import pandas as pd

def calcularRFM(df):
    fecha_referencia = df["InvoiceDate"].max() + pd.Timedelta(days=1)
    print(f"Fecha de referencia para la Recencia: {fecha_referencia}")

    df["ImporteTotal"] = df["Quantity"] * df["Price"]

    # Agrupo por cliente y calculo cada métrica por separado
    ultima_compra = df.groupby("Customer ID")["InvoiceDate"].max()
    frecuencia = df.groupby("Customer ID")["Invoice"].nunique()
    monto = df.groupby("Customer ID")["ImporteTotal"].sum()

    rfm = pd.DataFrame({
        "Customer ID": ultima_compra.index,
        "UltimaCompra": ultima_compra.values,
        "Frecuencia": frecuencia.values,
        "Monto": monto.values
    })

    # Cálculo de la recencia en días
    rfm["Recencia"] = (fecha_referencia - rfm["UltimaCompra"]).dt.days

    rfm = rfm.drop(columns="UltimaCompra")
    rfm = rfm.reset_index(drop=True)

    return rfm


def resumenRFM(rfm):

    print(rfm.head())

    print("\n Estadísticas descriptivas del RFM")
    print(rfm[["Recencia", "Frecuencia", "Monto"]].describe())

    print(f"\nCantidad de clientes en el RFM: {len(rfm)}")