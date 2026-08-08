from sklearn.preprocessing import StandardScaler
import pandas as pd

def escalarDatos(datos):

    columnas = ["Recencia", "Frecuencia", "Monto"]
    valores = datos[columnas]

    scaler = StandardScaler()
    valores_escalados = scaler.fit_transform(valores.values)

    df_escalado = pd.DataFrame(valores_escalados, columns=columnas)

    return df_escalado, scaler



def resumenEscalamiento(rfm_escalado):

    print(rfm_escalado.head())

    print("\n Estadísticas descriptivas después de escalar")
    print(rfm_escalado[["Recencia", "Frecuencia", "Monto"]].describe())