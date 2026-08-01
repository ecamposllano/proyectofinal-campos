def resumenInicial(df):

    print("\n--- Primeras 5 filas ---")
    print(df.head())

    print("\n--- Informacion general ---")
    print(df.info())

    print("\n--- Datos estadisticos ---")
    print(df.describe())

    print("\n--- Nulos por columna ---")
    print(df.isnull().sum())

    print("\n--- Duplicados ---")
    print(df.duplicated().sum())

    print("\n--- Cancelaciones ---")
    print(df["Invoice"].astype(str).str.startswith("C").sum())

    print("\n--- Rango de fechas ---")
    print(df["InvoiceDate"].min(), "a", df["InvoiceDate"].max())

    print("\n--- Clientes únicos ---")
    print(df["Customer ID"].nunique())

    print("\n--- Paises ---")
    print(df["Country"].unique())