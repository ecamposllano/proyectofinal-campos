def limpiarDatos(df):

    n_inicial = len(df)
    df_limpio = df.copy()

    # Sin Customer ID
    df_limpio = df_limpio.dropna(subset=["Customer ID"])

    # Cancelaciones (Invoice empieza con "C")
    df_limpio = df_limpio[~df_limpio["Invoice"].astype(str).str.startswith("C")]

    # Quantity y Price inválidos
    df_limpio = df_limpio[(df_limpio["Quantity"] > 0) & (df_limpio["Price"] > 0)]

    # 4. Duplicados exactos
    df_limpio = df_limpio.drop_duplicates()

    n_final = len(df_limpio)
    print(f"Filas antes de limpiar: {n_inicial}")
    print(f"Filas después de limpiar: {n_final}")
    print(f"Porcentaje eliminado: {(1 - n_final / n_inicial) * 100:.1f}%")
    print(f"Clientes únicos (después de limpiar): {df_limpio['Customer ID'].nunique()}")

    return df_limpio


def resumenLimpieza(df_original, df_limpio):

    sin_customer_id = df_original["Customer ID"].isna().sum()
    cancelaciones = df_original["Invoice"].astype(str).str.startswith("C").sum()
    cantidad_invalida = (df_original["Quantity"] <= 0).sum()
    precio_invalido = (df_original["Price"] <= 0).sum()

    print("--- Motivos de eliminación (no excluyentes entre sí) ---")
    print(f"Filas sin Customer ID: {sin_customer_id}")
    print(f"Filas de cancelaciones (Invoice con 'C'): {cancelaciones}")
    print(f"Filas con Quantity negativo: {cantidad_invalida}")
    print(f"Filas con Price negativo: {precio_invalido}")
    print(f"\nTotal filas originales: {len(df_original)}")
    print(f"Total filas limpias: {len(df_limpio)}")
