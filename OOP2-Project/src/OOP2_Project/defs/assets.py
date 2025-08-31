import io
from dagster import (
    asset, 
    AssetExecutionContext,
    AutoMaterializePolicy,
    Failure,
    MetadataValue,
    TableRecord,
    TableSchema,
    TableColumn,
    asset_check,
    AssetCheckResult
)
import pandas as pd
import requests

URL = "https://catalog.ourworldindata.org/garden/covid/latest/compact/compact.csv"
COUNTRIES = ["Ecuador", "Vietnam"]

# ---------------------------
# Paso 1: Lectura de datos
# ---------------------------
@asset(
    description="Descarga datos de la fuente canónica (CSV)",
    auto_materialize_policy=AutoMaterializePolicy.eager()
)
def covid_data(context: AssetExecutionContext) -> pd.DataFrame:
    try:
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        context.log.info(f"Datos descargados: {len(df)} filas, {len(df.columns)} columnas")
        return df
    except requests.RequestException as e:
        raise Failure(description="Error al descargar datos", metadata={"url": MetadataValue.text(URL), "error": MetadataValue.text(str(e))})
    except KeyError as e:
        raise Failure(description="Error al procesar datos", metadata={"url": MetadataValue.text(URL), "error": MetadataValue.text(str(e))})

# ---------------------------
# Paso 2: Chequeos de entrada
# ---------------------------
@asset_check(asset=covid_data, description="Valida que no haya valores nulos", blocking=False)
def check_columnas_no_nulas(covid_data: pd.DataFrame) -> AssetCheckResult:
    columnas_clave = ["country", "date", "population"]
    filas_afectadas = covid_data[covid_data[columnas_clave].isnull().any(axis=1)]
    estado = len(filas_afectadas) == 0
    notas = "OK" if estado else f"Nulos detectados: {len(filas_afectadas)}"
    return AssetCheckResult(passed=estado, metadata={"filas_afectadas": len(filas_afectadas), "notas": notas})

@asset_check(asset=covid_data, description="Valida unicidad de country y date", blocking=False)
def check_unicidad(covid_data: pd.DataFrame) -> AssetCheckResult:
    filas_afectadas = covid_data[covid_data.duplicated(subset=["country", "date"], keep=False)]
    estado = len(filas_afectadas) == 0
    notas = "OK" if estado else f"Duplicados detectados: {len(filas_afectadas)}"
    return AssetCheckResult(passed=estado, metadata={"filas_afectadas": len(filas_afectadas), "notas": notas})

# ---------------------------
# Paso 3: Datos procesados
# ---------------------------
@asset(
    description="Datos procesados y filtrados para análisis",
    group_name="procesamiento",
    auto_materialize_policy=AutoMaterializePolicy.eager()
)
def datos_procesados(context: AssetExecutionContext, covid_data: pd.DataFrame) -> pd.DataFrame:
    df = covid_data.copy()
    df = df[df["country"].isin(COUNTRIES)]
    df = df.dropna(subset=["new_cases", "people_vaccinated"])
    df = df.drop_duplicates(subset=["country", "date"])
    return df[["country", "date", "new_cases", "people_vaccinated", "population"]]

# ---------------------------
# Paso 4A: Incidencia 7 días
# ---------------------------
@asset(
    description="Métrica de incidencia acumulada a 7 días por 100mil habitantes",
    group_name="metric_calcs",
    auto_materialize_policy=AutoMaterializePolicy.eager()
)
def metrica_incidencia_7d(datos_procesados: pd.DataFrame) -> pd.DataFrame:
    df = datos_procesados.copy()
    df["incidencia_diaria"] = (df["new_cases"] / df["population"]) * 100000
    df["incidencia_7d"] = df.groupby("country")["incidencia_diaria"].transform(lambda x: x.rolling(7).mean())
    return df[["date", "country", "incidencia_7d"]]

# ---------------------------
# Paso 4B: Factor de crecimiento semanal
# ---------------------------
@asset(
    description="Métrica de factor de crecimiento semanal de casos",
    group_name="metric_calcs",
    auto_materialize_policy=AutoMaterializePolicy.eager()
)
def metrica_factor_crec_7d(datos_procesados: pd.DataFrame) -> pd.DataFrame:
    df = datos_procesados.copy()
    resultados = []
    for pais, grupo in df.groupby("country"):
        grupo = grupo.sort_values("date")
        grupo["casos_semana_actual"] = grupo["new_cases"].rolling(7, min_periods=7).sum()
        grupo["casos_semana_prev"] = grupo["new_cases"].shift(7).rolling(7, min_periods=7).sum()
        grupo["factor_crec_7d"] = grupo["casos_semana_actual"] / grupo["casos_semana_prev"].replace({0: pd.NA})
        resultados.append(grupo[["date", "country", "casos_semana_actual", "factor_crec_7d"]])
    final = pd.concat(resultados)
    final = final.rename(columns={"country": "país", "date": "semana_fin", "casos_semana_actual": "casos_semana"})
    return final.dropna(subset=["factor_crec_7d"])

# ---------------------------
# Paso 5: Chequeos de salida
# ---------------------------
@asset_check(
    asset="metrica_factor_crec_7d",
    description="Verifica que factor de crecimiento ≤ 10",
    blocking=False
)
def check_factor_crec_irreal(metrica_factor_crec_7d: pd.DataFrame) -> AssetCheckResult:
    irreales = metrica_factor_crec_7d[metrica_factor_crec_7d["factor_crec_7d"] > 10]
    schema = TableSchema(columns=[
        TableColumn(name="semana_fin", type="string"),
        TableColumn(name="país", type="string"),
        TableColumn(name="casos_semana", type="int"),
        TableColumn(name="factor_crec_7d", type="float"),
    ])
    table_records = [TableRecord(**row) for _, row in irreales.iterrows()]
    return AssetCheckResult(passed=irreales.empty, metadata={"factores_irreales": MetadataValue.table(records=table_records, schema=schema), "total_irreales": len(irreales)})

@asset_check(
    asset="metrica_incidencia_7d",
    description="Chequea que 0 ≤ incidencia_7d ≤ 2000"
)
def check_incidencia_valores(metrica_incidencia_7d: pd.DataFrame) -> AssetCheckResult:
    fuera_rango = metrica_incidencia_7d[(metrica_incidencia_7d["incidencia_7d"] < 0) | (metrica_incidencia_7d["incidencia_7d"] > 2000)]
    schema = TableSchema(columns=[
        TableColumn(name="date", type="string"),
        TableColumn(name="country", type="string"),
        TableColumn(name="incidencia_7d", type="float")
    ])
    table_records = [TableRecord(**row) for _, row in fuera_rango.iterrows()]
    return AssetCheckResult(passed=fuera_rango.empty, metadata={"fuera_rango": MetadataValue.table(records=table_records, schema=schema), "total_fuera_rango": len(fuera_rango)})

# ---------------------------
# Paso 6: Exportación
# ---------------------------
@asset(
    description="Exporta resultados finales a Excel",
    group_name="reportes",
)
def reporte(datos_procesados: pd.DataFrame, metrica_incidencia_7d: pd.DataFrame, metrica_factor_crec_7d: pd.DataFrame):
    filename = "covid_reporte.xlsx"
    with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
        datos_procesados.to_excel(writer, sheet_name="Datos_Procesados_Limpios", index=False)
        metrica_incidencia_7d.to_excel(writer, sheet_name="Incidencia7d", index=False)
        metrica_factor_crec_7d.to_excel(writer, sheet_name="FactorCrecimiento", index=False)
    return filename
