import os
import logging
import warnings
import json
import joblib
import math
import pandas as pd
from pandas.api.types import is_numeric_dtype
import numpy as np
import statistics
import shap
from typing import Any, Dict, List, Union, Tuple
import re

from utils.data_objects import (
    taxonomic_dict
)

# Suppress TensorFlow and other verbose logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'         # FATAL only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'       # disable oneDNN custom ops
try:
    # absl is used by TensorFlow
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass
logging.getLogger('tensorflow').setLevel(logging.ERROR)
# Suppress scikit-learn version mismatch warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings('ignore', category=InconsistentVersionWarning)
# Suppress pandas duplicate column warnings
warnings.filterwarnings('ignore', message=".*DataFrame columns are not unique.*")

from tensorflow.keras.models import load_model

# Initialize file paths for scaler, means CSV, and PCA model
scaler_path = r"models\PCA\scaler_minmax.pkl"
means_csv_path = r"models\PCA\mean_scale.csv"
pca_path = r"models\PCA\pca_model.pkl"
pruned_clf_path = r"models\DT\pruned_clf.joblib"
nn_path = r"models\NN\nn_model.h5"


#EDA processor--------------------------------------------------------------
def transform_GENEROBIN_ORIENTSEXBN(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores faltantes en las columnas 'GENERO_BIN' y 'ORIENTSEX.BN'.

    Reglas de imputación:
      - En 'GENERO_BIN', reemplaza NaN con 2.
      - En 'ORIENTSEX.BN', reemplaza NaN con 3.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que debe contener las columnas 'GENERO_BIN' y 'ORIENTSEX.BN'.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame original con las dos columnas imputadas y convertidas a entero.

    Lanza
    -----
    KeyError
        Si falta alguna de las columnas requeridas.
    """
    # Copiamos el DataFrame para no modificar el original
    df_transformed = df.copy()

    required_cols = ["GENERO_BIN", "ORIENTSEX.BN"]
    missing = [col for col in required_cols if col not in df_transformed.columns]
    if missing:
        raise KeyError(f"Faltan las columnas requeridas: {missing}")

    # Imputación
    df_transformed["GENERO_BIN"]    = df_transformed["GENERO_BIN"].fillna(2).astype(int)
    df_transformed["ORIENTSEX.BN"] = df_transformed["ORIENTSEX.BN"].fillna(3).astype(int)

    return df_transformed
def transform_PPBULL(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la columna 'PP.BULL' codificada según la presencia de bullying en PP1.BULL o PP2.BULL:

      - 1: si alguna de las columnas 'PP1.BULL' o 'PP2.BULL' es igual a 1.
      - 0: en cualquier otro caso (incluye valores 0 o NaN en ambas columnas).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que debe contener las columnas 'PP1.BULL' y 'PP2.BULL'.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame original con la nueva columna 'PP.BULL'.

    Lanza
    -----
    KeyError
        Si faltan las columnas requeridas.
    """
    # Trabajar sobre copia para no modificar el original
    df_transformed = df.copy()

    required = ["PP1.BULL", "PP2.BULL"]
    missing = [col for col in required if col not in df_transformed.columns]
    if missing:
        raise KeyError(f"Faltan las columnas requeridas: {missing}")

    # Crear PP.BULL: 1 si alguna es 1, sino 0 (NaN → False)
    df_transformed["PP.BULL"] = (
        df_transformed[required]
        .eq(1)
        .any(axis=1)
        .astype(int)
    )

    return df_transformed
def transform_ABUSOSUBS(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma las columnas 'ABUSOSUBS1' y 'ABUSOSUBS2' de la siguiente manera:

      1. Convierte todos los valores iguales a 1 en 0.
      2. Imputa los valores faltantes (NaN) con 1 una vez aplicada la conversión anterior.
      3. Asegura que el tipo final de la columna sea entero (int64).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada que debe contener las columnas
        'ABUSOSUBS1' y 'ABUSOSUBS2'.

    Devuelve
    -------
    pd.DataFrame
        Una copia del DataFrame original con las columnas transformadas.

    Lanza
    -----
    KeyError
        Si faltan una o ambas columnas requeridas en el DataFrame.
    """
    # Trabajamos sobre una copia para no mutar el df original
    df_transformed = df.copy()
    cols = ["ABUSOSUBS1", "ABUSOSUBS2"]

    # Verificar que existan las columnas necesarias
    missing = [c for c in cols if c not in df_transformed.columns]
    if missing:
        raise KeyError(f"Faltan las columnas requeridas: {missing}")

    for col in cols:
        # 1) Convertir 1 → 0
        df_transformed[col] = df_transformed[col].replace(1, 0)
        # 2) Rellenar NaN → 1
        df_transformed[col] = df_transformed[col].fillna(1)
        # 3) Convertir a entero
        df_transformed[col] = df_transformed[col].astype(int)

    return df_transformed
def transform_AUTOEFIC(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores faltantes en las columnas 'AUTOEFIC1' a 'AUTOEFIC5' con la media de cada columna,
    y añade dos nuevas columnas:
      - 'AUTOEFIC.MEAN': media por fila de los cinco ítems imputados.
      - 'AUTOEFIC.VAR' : varianza por fila de los cinco ítems imputados (varianza poblacional, ddof=0).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada que debe contener las columnas
        'AUTOEFIC1', 'AUTOEFIC2', 'AUTOEFIC3', 'AUTOEFIC4', 'AUTOEFIC5'.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame original con:
        - Las columnas 'AUTOEFIC1'...'AUTOEFIC5' imputadas sin NaN.
        - Dos columnas nuevas: 'AUTOEFIC.MEAN' y 'AUTOEFIC.VAR'.

    Lanza
    -----
    KeyError
        Si falta alguna de las columnas requeridas.
    """
    # Hacemos una copia para no modificar el df original
    df_transformed = df.copy()
    cols = [f"AUTOEFIC{i}" for i in range(1, 6)]

    # Verificar existencia de columnas
    missing = [c for c in cols if c not in df_transformed.columns]
    if missing:
        raise KeyError(f"Faltan las columnas requeridas: {missing}")

    # Imputación: llenar NaN con la media de cada columna
    for col in cols:
        mean_val = df_transformed[col].mean(skipna=True)
        df_transformed[col] = df_transformed[col].fillna(mean_val)

    # Cálculo de la media y varianza por fila
    df_transformed["AUTOEFIC.MEAN"] = df_transformed[cols].mean(axis=1)
    df_transformed["AUTOEFIC.VAR"]  = df_transformed[cols].var(axis=1, ddof=0)

    return df_transformed
def transform_CONVIVEN_beta(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajusta la columna 'CONVIVEN.6' sumándole el valor de 'CONVIVEN.7' y luego
    capea todos los valores resultantes a un máximo de 1.

    Pasos:
      1. Verifica que existan las columnas 'CONVIVEN.1' … 'CONVIVEN.7'.
      2. Suma, para cada fila, CONVIVEN.6 + CONVIVEN.7 (tratando NaN como 0).
      3. Si el resultado de la suma es mayor que 1, lo deja en 1.
      4. Convierte la columna resultante a entero.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada con las columnas 'CONVIVEN.1' … 'CONVIVEN.7'.

    Devuelve
    -------
    pd.DataFrame
        Copia del DataFrame con 'CONVIVEN.6' actualizada y capada a 1.
        Las demás columnas permanecen iguales.

    Lanza
    -----
    KeyError
        Si falta alguna de las columnas necesarias.
    """
    df_transformed = df.copy()
    df_transformed["CONVIVEN.6"].fillna(0)
    df_transformed = df_transformed.drop(columns=["CONVIVEN.7"])

    return df_transformed
def transform_IMPULS(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa los valores faltantes en las columnas de impulsividad y añade tres
    estadísticas por fila: media, varianza y mediana.

    Pasos:
      1. Verifica que existan las columnas:
         'IMPULS1', 'IMPULS2_REV', 'IMPULS3.REV', 'IMPULS4',
         'IMPULS5', 'IMPULS6', 'IMPULS7.REV', 'IMPULS8.REV'.
      2. Para cada una, imputa los NaN con la media de su propia columna.
      3. Crea una nueva columna 'IMPULS.MEAN' con la media por fila de las 8 variables.
      4. Crea una nueva columna 'IMPULS.VAR' con la varianza poblacional (ddof=0) por fila.
      5. Crea una nueva columna 'IMPULS.MEDIAN' con la mediana por fila.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene las columnas de impulsividad.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame original con las columnas imputadas y las tres
        estadísticas añadidas.

    Lanza
    -----
    KeyError
        Si faltan columnas requeridas.
    """
    # Hacemos una copia para no mutar el original
    df_transformed = df.copy()

    cols = [
        "IMPULS1", "IMPULS2_REV", "IMPULS3.REV", "IMPULS4",
        "IMPULS5", "IMPULS6", "IMPULS7.REV", "IMPULS8.REV"
    ]

    # Verificación de columnas
    missing = [c for c in cols if c not in df_transformed.columns]
    if missing:
        raise KeyError(f"Faltan las columnas requeridas: {missing}")

    # Imputación de NaN con la media de cada columna
    for col in cols:
        mean_val = df_transformed[col].mean(skipna=True)
        df_transformed[col] = df_transformed[col].fillna(mean_val)

    # Cálculo de estadísticas por fila
    df_transformed["IMPULS.MEAN"]   = df_transformed[cols].mean(axis=1)
    df_transformed["IMPULS.VAR"]    = df_transformed[cols].var(axis=1, ddof=0)
    df_transformed["IMPULS.MEDIAN"] = df_transformed[cols].median(axis=1)

    return df_transformed
def transform_APOYO(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores faltantes en las columnas de apoyo y añade tres
    estadísticas por fila: media, varianza y mediana.

    Pasos:
      1. Verifica que existan las columnas:
         'APOYO1', 'APOYO2', 'APOYO3', 'APOYO4',
         'APOYO5', 'APOYO6', 'APOYO7'.
      2. Para cada una, imputa los NaN con la media de su propia columna.
      3. Crea una nueva columna 'APOYO.MEAN' con la media por fila de las 7 variables.
      4. Crea una nueva columna 'APOYO.VAR' con la varianza poblacional (ddof=0) por fila.
      5. Crea una nueva columna 'APOYO.MEDIAN' con la mediana por fila.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene las columnas de apoyo.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame original con las columnas imputadas y las tres
        estadísticas añadidas.

    Lanza
    -----
    KeyError
        Si faltan columnas requeridas.
    """
    # Hacemos una copia para no mutar el original
    df_transformed = df.copy()

    cols = [f"APOYO{i}" for i in range(1, 8)]

    # Verificación de columnas
    missing = [c for c in cols if c not in df_transformed.columns]
    if missing:
        raise KeyError(f"Faltan las columnas requeridas: {missing}")

    # Imputación de NaN con la media de cada columna
    for col in cols:
        mean_val = df_transformed[col].mean(skipna=True)
        df_transformed[col] = df_transformed[col].fillna(mean_val)

    # Cálculo de estadísticas por fila
    df_transformed["APOYO.MEAN"]   = df_transformed[cols].mean(axis=1)
    df_transformed["APOYO.VAR"]    = df_transformed[cols].var(axis=1, ddof=0)
    df_transformed["APOYO.MEDIAN"] = df_transformed[cols].median(axis=1)

    return df_transformed
def transform_VEXPQUIEN(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea una nueva columna 'VEXP.QUIEN' codificada según estas reglas:

      - 1: desconocido  (alguna de las columnas originales contiene un 1)
      - 0: conocido      (ninguna contiene un 1, y al menos un valor no es NaN)
      - 2: nadie         (todas las columnas originales son NaN)

    Pasos:
      1. Verifica que existan las columnas
         'VEXP1.QUIEN.1', 'VEXP2.QUIEN.1' y 'VEXP3.QUIEN.1'.
      2. Para cada fila, comprueba:
         a) Si todas las columnas son NaN → asigna 2.
         b) Si alguna columna == 1 → asigna 1.
         c) En caso contrario → asigna 0.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que debe contener las columnas
        'VEXP1.QUIEN.1', 'VEXP2.QUIEN.1', 'VEXP3.QUIEN.1'.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame original con la nueva columna 'VEXP.QUIEN'.

    Lanza
    -----
    KeyError
        Si falta alguna de las columnas requeridas.
    """
    df_transformed = df.copy()
    cols = ["VEXP1.QUIEN.1", "VEXP2.QUIEN.1", "VEXP3.QUIEN.1"]

    # Validación de columnas
    missing = [c for c in cols if c not in df_transformed.columns]
    if missing:
        raise KeyError(f"Faltan las columnas requeridas: {missing}")

    # Identificar casos
    all_nan    = df_transformed[cols].isna().all(axis=1)
    has_one    = df_transformed[cols].eq(1).any(axis=1)

    # Asignar por defecto 'conocido' (0)
    df_transformed["VEXP.QUIEN"] = 0
    # Luego sobreescribir desconocido (1)
    df_transformed.loc[has_one, "VEXP.QUIEN"] = 1
    # Finalmente, nadie (2) donde todas son NaN
    df_transformed.loc[all_nan, "VEXP.QUIEN"] = 2

    return df_transformed
def transform_VEXPCONTACT(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la columna 'VEXP.CONTACT' con codificación según el tipo de contacto,
    a partir de las columnas binarias de contacto en persona y virtual para
    VEXP1, VEXP2 y VEXP3.

    Codificación de 'VEXP.CONTACT':
      - 3 : nadie (todas las columnas de contacto son NaN)
      - 0 : presencial (al menos un 1 en las columnas de contacto presenciales)
      - 1 : virtual   (al menos un 1 en las columnas de contacto virtuales)
      - 2 : ambos      (al menos un 1 en presenciales y al menos un 1 en virtuales)

    Columnas de entrada esperadas:
      Presenciales:
        VEXP1.CONTACT.1  VEXP1.CONTACT.2  VEXP1.CONTACT.3
        VEXP2.CONTACT.1  VEXP2.CONTACT.2  VEXP2.CONTACT.3
        VEXP3.CONTACT.1  VEXP3.CONTACT.2  VEXP3.CONTACT.3

      Virtuales:
        VEXP1.CONTACT.4  ...  VEXP1.CONTACT.12
        VEXP2.CONTACT.4  ...  VEXP2.CONTACT.12
        VEXP3.CONTACT.4  ...  VEXP3.CONTACT.12

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene todas las columnas 'VEXP{1,2,3}.CONTACT.{1..12}'.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame original con la columna adicional 'VEXP.CONTACT'.

    Lanza
    -----
    KeyError
        Si falta alguna de las columnas requeridas.
    """
    df_transformed = df.copy()

    # Generar listas de nombres de columna
    pres_cols = [
        f"VEXP{i}.CONTACT.{j}"
        for i in range(1, 4)
        for j in range(1, 4)
    ]
    virt_cols = [
        f"VEXP{i}.CONTACT.{j}"
        for i in range(1, 4)
        for j in range(4, 13)
    ]
    all_cols = pres_cols + virt_cols

    # Validar que existan todas las columnas
    missing = [c for c in all_cols if c not in df_transformed.columns]
    if missing:
        raise KeyError(f"Faltan las columnas requeridas: {missing}")

    # Identificar patrones por fila
    all_nan    = df_transformed[all_cols].isna().all(axis=1)
    has_pres   = df_transformed[pres_cols].eq(1).any(axis=1)
    has_virt   = df_transformed[virt_cols].eq(1).any(axis=1)

    # Inicializar con 0 (presencial)
    df_transformed["VEXP.CONTACT"] = 0

    # Ambos (virtual + presencial)
    df_transformed.loc[has_pres & has_virt, "VEXP.CONTACT"] = 2
    # Virtual únicamente
    df_transformed.loc[~has_pres & has_virt, "VEXP.CONTACT"] = 1
    # Nadie (todas NaN)
    df_transformed.loc[all_nan, "VEXP.CONTACT"] = 3

    return df_transformed
def transform_VMQUIEN(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la columna 'VM.QUIEN' codificada según la presencia de respuestas
    en las columnas VM{1..4}.QUIEN.{1..5}.

    Codificación de 'VM.QUIEN':
      - 1 : padres          (alguna de VM{i}.QUIEN.1 o VM{i}.QUIEN.2 es 1)
      - 0 : no padres       (alguna de VM{i}.QUIEN.3, VM{i}.QUIEN.4 o VM{i}.QUIEN.5 es 1,
                             y ninguna de las de 'padres' es 1)
      - 2 : ambos           (al menos un 1 en grupo 'padres' y al menos un 1 en grupo 'no padres')
      - 3 : nadie (todas NaN en las 20 columnas)

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que debe contener las columnas:
          VM1.QUIEN.1 … VM1.QUIEN.5,
          VM2.QUIEN.1 … VM2.QUIEN.5,
          VM3.QUIEN.1 … VM3.QUIEN.5,
          VM4.QUIEN.1 … VM4.QUIEN.5.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame original con la nueva columna 'VM.QUIEN'.

    Lanza
    -----
    KeyError
        Si falta alguna de las columnas requeridas.
    """
    df_transformed = df.copy()

    # Columnas de "padres"
    parent_cols = [
        f"VM{i}.QUIEN.{j}"
        for i in range(1, 5)
        for j in (1, 2)
    ]
    # Columnas de "no padres"
    non_parent_cols = [
        f"VM{i}.QUIEN.{j}"
        for i in range(1, 5)
        for j in (3, 4, 5)
    ]
    all_cols = parent_cols + non_parent_cols

    # Validar existencia de todas las columnas
    missing = [c for c in all_cols if c not in df_transformed.columns]
    if missing:
        raise KeyError(f"Faltan las columnas requeridas: {missing}")

    # Flags por fila
    all_nan       = df_transformed[all_cols].isna().all(axis=1)
    has_parent    = df_transformed[parent_cols].eq(1).any(axis=1)
    has_non_par   = df_transformed[non_parent_cols].eq(1).any(axis=1)

    # Inicializamos con 0 ("no padres")
    df_transformed["VM.QUIEN"] = 0

    # 2 = ambos
    df_transformed.loc[has_parent & has_non_par, "VM.QUIEN"] = 2
    # 1 = padres solamente
    df_transformed.loc[has_parent & ~has_non_par, "VM.QUIEN"] = 1
    # 0 = no padres solamente (ya es 0 por defecto, se mantiene)
    # 3 = nadie (todas NaN)
    df_transformed.loc[all_nan, "VM.QUIEN"] = 3

    return df_transformed
def transform_WQUIEN(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la columna 'W.QUIEN' codificada según respuestas en las columnas
    W{1..4}.QUIEN.{1..4}.

    Codificación de 'W.QUIEN':
      - 3 : Nadie (todas las columnas son NaN)
      - 1 : Padres (al menos un 1 en las preguntas .1 o .2 de cualquiera de W1–W4,
                  y ninguno en .3 o .4)
      - 0 : No padres (al menos un 1 en las preguntas .3 o .4 de cualquiera de W1–W4,
                      y ninguno en .1 o .2)
      - 2 : Ambos (al menos un 1 en .1/.2 y al menos un 1 en .3/.4)

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene las siguientes columnas:
          W1.QUIEN.1  W1.QUIEN.2  W1.QUIEN.3  W1.QUIEN.4
          W2.QUIEN.1  W2.QUIEN.2  W2.QUIEN.3  W2.QUIEN.4
          W3.QUIEN.1  W3.QUIEN.2  W3.QUIEN.3  W3.QUIEN.4
          W4.QUIEN.1  W4.QUIEN.2  W4.QUIEN.3  W4.QUIEN.4

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame original con la nueva columna 'W.QUIEN'.

    Lanza
    -----
    KeyError
        Si faltan columnas requeridas.
    """
    df_out = df.copy()

    # Columnas relacionadas con "padres" (.1 y .2) y "no padres" (.3 y .4)
    parent_cols = [
        f"W{i}.QUIEN.{j}"
        for i in range(1, 5)
        for j in (1, 2)
    ]
    non_parent_cols = [
        f"W{i}.QUIEN.{j}"
        for i in range(1, 5)
        for j in (3, 4)
    ]
    all_cols = parent_cols + non_parent_cols

    # Validar existencia de columnas
    missing = [c for c in all_cols if c not in df_out.columns]
    if missing:
        raise KeyError(f"Faltan las columnas requeridas: {missing}")

    # Flags por fila
    all_nan    = df_out[all_cols].isna().all(axis=1)
    has_parent = df_out[parent_cols].eq(1).any(axis=1)
    has_nonpar = df_out[non_parent_cols].eq(1).any(axis=1)

    # Inicializamos con 0 ("No padres")
    df_out["W.QUIEN"] = 0

    # 2 = Ambos (tienen al menos un 1 en ambos grupos)
    df_out.loc[has_parent & has_nonpar, "W.QUIEN"] = 2
    # 1 = Padres únicamente
    df_out.loc[has_parent & ~has_nonpar, "W.QUIEN"] = 1
    # 0 = No padres únicamente (ya es 0 por defecto)
    # 3 = Nadie (todas NaN)
    df_out.loc[all_nan, "W.QUIEN"] = 3

    return df_out
def transform_PAIS(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa los valores faltantes en la columna 'PAÍS' con el valor 2.

    Pasos:
      1. Verifica que exista la columna 'PAÍS'.
      2. Rellena los NaN de 'PAÍS' con 2.
      3. Convierte la columna resultante a entero si procede.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que debe contener la columna 'PAÍS'.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame original con la columna 'PAÍS' imputada.

    Lanza
    -----
    KeyError
        Si falta la columna 'PAÍS'.
    """
    # Copiar para no modificar el original
    df_transformed = df.copy()

    # Verificar existencia de la columna
    if "PAÍS" not in df_transformed.columns:
        raise KeyError("Falta la columna requerida: 'PAÍS'")

    # Imputar NaN con 2 y convertir a entero
    df_transformed["PAÍS"] = df_transformed["PAÍS"].fillna(2).astype(int)

    return df_transformed
def transform_MORAL(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa valores faltantes en las columnas 'MORAL1' a 'MORAL5' con la media de cada columna,
    y añade dos nuevas columnas:
      - 'MORAL.MEAN': media por fila de los cinco ítems imputados.
      - 'MORAL.VAR' : varianza por fila de los cinco ítems imputados (varianza poblacional, ddof=0).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame de entrada que debe contener las columnas
        'MORAL1', 'MORAL2', 'MORAL3', 'MORAL4', 'MORAL5'.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame original con:
        - Las columnas 'MORAL1'...'MORAL5' imputadas sin NaN.
        - Dos columnas nuevas: 'MORAL.MEAN' y 'MORAL.VAR'.

    Lanza
    -----
    KeyError
        Si falta alguna de las columnas requeridas.
    """
    # Trabajar sobre copia para no mutar el df original
    df_transformed = df.copy()
    cols = [f"MORAL{i}" for i in range(1, 6)]

    # Verificar existencia de columnas
    missing = [c for c in cols if c not in df_transformed.columns]
    if missing:
        raise KeyError(f"Faltan las columnas requeridas: {missing}")

    # Imputación: llenar NaN con la media de cada columna
    for col in cols:
        mean_val = df_transformed[col].mean(skipna=True)
        df_transformed[col] = df_transformed[col].fillna(mean_val)

    # Cálculo de la media y varianza por fila
    df_transformed["MORAL.MEAN"] = df_transformed[cols].mean(axis=1)
    df_transformed["MORAL.VAR"]  = df_transformed[cols].var(axis=1, ddof=0)

    return df_transformed
def transform_PEXPCONTACT(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la columna 'PEXP.CONTACT' codificada según el tipo de contacto presencial
    o virtual en la única serie PEXP1.CONTACT.{1..12}.

    Codificación de 'PEXP.CONTACT':
      - 3 : nadie       (todas las columnas PEXP1.CONTACT.1–12 son NaN)
      - 1 : presencial  (al menos un 1 en PEXP1.CONTACT.1–3, y ninguno en PEXP1.CONTACT.4–12)
      - 0 : virtual     (al menos un 1 en PEXP1.CONTACT.4–12, y ninguno en PEXP1.CONTACT.1–3)
      - 2 : ambos       (al menos un 1 en PEXP1.CONTACT.1–3 y al menos un 1 en PEXP1.CONTACT.4–12)

    Pasos
    -----
      1. Verifica que existan las columnas PEXP1.CONTACT.1 … PEXP1.CONTACT.12.
      2. Calcula:
         - `has_pres` = alguna de PEXP1.CONTACT.1–3 == 1
         - `has_virt` = alguna de PEXP1.CONTACT.4–12 == 1
         - `all_nan`  = todas PEXP1.CONTACT.1–12 son NaN
      3. Inicializa 'PEXP.CONTACT' = 0 (virtual).
      4. Asigna 2 donde `has_pres & has_virt` (ambos).
      5. Asigna 1 donde `has_pres & ~has_virt` (presencial únicamente).
      6. Asigna 3 donde `all_nan` (nadie).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene las columnas PEXP1.CONTACT.1 … PEXP1.CONTACT.12.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame original con la nueva columna 'PEXP.CONTACT'.

    Lanza
    -----
    KeyError
        Si falta alguna de las columnas esperadas.
    """
    # Copiar para no mutar el original
    df_transformed = df.copy()

    # Definir listas de columnas
    pres_cols = [f"PEXP1.CONTACT.{i}" for i in (1, 2, 3)]
    virt_cols = [f"PEXP1.CONTACT.{i}" for i in range(4, 13)]
    all_cols = pres_cols + virt_cols

    # Verificar existencia de todas las columnas
    missing = [c for c in all_cols if c not in df_transformed.columns]
    if missing:
        raise KeyError(f"Faltan las columnas requeridas: {missing}")

    # Flags por fila
    has_pres = df_transformed[pres_cols].eq(1).any(axis=1)
    has_virt = df_transformed[virt_cols].eq(1).any(axis=1)
    all_nan  = df_transformed[all_cols].isna().all(axis=1)

    # Inicializar como 'virtual' (0)
    df_transformed["PEXP.CONTACT"] = 0

    # Ambos (2)
    df_transformed.loc[has_pres & has_virt, "PEXP.CONTACT"] = 2
    # Presencial únicamente (1)
    df_transformed.loc[has_pres & ~has_virt, "PEXP.CONTACT"] = 1
    # Nadie (3)
    df_transformed.loc[all_nan, "PEXP.CONTACT"] = 3

    return df_transformed
def transform_VSQUIEN(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la columna 'VS.QUIEN' codificada según presencia de respuestas en las
    columnas VS1.QUIEN.1–7, VS2.QUIEN.1–7, VS5.QUIEN.1–4 y VS6.QUIEN.1–4.

    Codificación de 'VS.QUIEN':
      - 3 : nadie (todas las columnas son NaN)
      - 1 : familia directa (al menos un 1 en VS1.QUIEN.1–2, VS2.QUIEN.1–2,
            VS5.QUIEN.2 o VS6.QUIEN.2, y ningún 1 fuera de ese grupo)
      - 0 : persona lejana (al menos un 1 en VS1.QUIEN.3–7, VS2.QUIEN.3–7,
            VS5.QUIEN.1,3–4 o VS6.QUIEN.1,3–4, y ningún 1 en el grupo 'familia directa')
      - 2 : ambos (hay al menos un 1 en ambos grupos 'familia directa' y
            'persona lejana')

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que contiene las columnas:
          VS1.QUIEN.1 … VS1.QUIEN.7,
          VS2.QUIEN.1 … VS2.QUIEN.7,
          VS5.QUIEN.1 … VS5.QUIEN.4,
          VS6.QUIEN.1 … VS6.QUIEN.4.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame original con la nueva columna 'VS.QUIEN'.

    Lanza
    -----
    KeyError
        Si falta alguna de las columnas requeridas.
    """
    df_out = df.copy()

    # Definir columnas de los dos grupos
    direct_cols = [
        "VS1.QUIEN.1", "VS1.QUIEN.2",
        "VS2.QUIEN.1", "VS2.QUIEN.2",
        "VS5.QUIEN.2",
        "VS6.QUIEN.2"
    ]
    remote_cols = [
        "VS1.QUIEN.3", "VS1.QUIEN.4", "VS1.QUIEN.5", "VS1.QUIEN.6", "VS1.QUIEN.7",
        "VS2.QUIEN.3", "VS2.QUIEN.4", "VS2.QUIEN.5", "VS2.QUIEN.6", "VS2.QUIEN.7",
        "VS5.QUIEN.1", "VS5.QUIEN.3", "VS5.QUIEN.4",
        "VS6.QUIEN.1", "VS6.QUIEN.3", "VS6.QUIEN.4"
    ]
    all_cols = direct_cols + remote_cols

    # Verificar que existan todas las columnas
    missing = [c for c in all_cols if c not in df_out.columns]
    if missing:
        raise KeyError(f"Faltan las columnas requeridas: {missing}")

    # Flags por fila
    all_nan    = df_out[all_cols].isna().all(axis=1)
    has_direct = df_out[direct_cols].eq(1).any(axis=1)
    has_remote = df_out[remote_cols].eq(1).any(axis=1)

    # Inicializar todo como 'nadie' (3)
    df_out["VS.QUIEN"] = 3

    # Ambos (2)
    df_out.loc[has_direct & has_remote, "VS.QUIEN"] = 2
    # Familia directa únicamente (1)
    df_out.loc[has_direct & ~has_remote & ~all_nan, "VS.QUIEN"] = 1
    # Persona lejana únicamente (0)
    df_out.loc[~has_direct & has_remote & ~all_nan, "VS.QUIEN"] = 0
    # Los que quedan (sin 1 alguno ni all_nan) permanecerán como 3 (nadie)

    return df_out
def transform_VPBULL(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la columna 'VP.BULL' codificada según la presencia de bullying en VP1.BULL o VP2.BULL:

      - 1: si alguna de las columnas 'VP1.BULL' o 'VP2.BULL' es igual a 1.
      - 0: en cualquier otro caso (incluye valores 0 o NaN en ambas columnas).

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame que debe contener las columnas 'VP1.BULL' y 'VP2.BULL'.

    Retorna
    -------
    pd.DataFrame
        Copia del DataFrame original con la nueva columna 'VP.BULL'.

    Lanza
    -----
    KeyError
        Si faltan las columnas requeridas.
    """
    # Trabajar sobre copia para no modificar el original
    df_transformed = df.copy()

    required = ["VP1.BULL", "VP2.BULL"]
    missing = [col for col in required if col not in df_transformed.columns]
    if missing:
        raise KeyError(f"Faltan las columnas requeridas: {missing}")

    # Crear VP.BULL: 1 si alguna es 1, sino 0 (NaN → False)
    df_transformed["VP.BULL"] = (
        df_transformed[required]
        .eq(1)
        .any(axis=1)
        .astype(int)
    )

    return df_transformed
def nan_imputer(df: pd.DataFrame) -> pd.DataFrame:
    df["POLIVICTIMIZACION"] = df["POLIVICTIMIZACION"].fillna(0)
    df["POLIPERPETRACION"] = df["POLIPERPETRACION"].fillna(0)
    df["PORNO.T"] = df["PORNO.T"].fillna(1)
    df["FUGAS.BN"] = df["FUGAS.BN"].fillna(0)
    return df
def get_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica one-hot encoding a las columnas indicadas de un DataFrame.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame original.
    columns : list[str]
        Lista con los nombres de las columnas a dummyficar.

    Devuelve
    -------
    pd.DataFrame
        Nuevo DataFrame con las columnas originales removidas y reemplazadas
        por sus variables dummy.
    """
    columns = ['GENERO_BIN','ORIENTSEX.BN','PEXP.CONTACT','VEXP.QUIEN','VEXP.CONTACT','VS.QUIEN','VM.QUIEN','W.QUIEN']
    # Validación de que las columnas existen
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Las siguientes columnas no existen en el DataFrame: {missing}")

    # Usamos pandas.get_dummies para todo el DataFrame
    df_encoded = pd.get_dummies(df, columns=columns, dtype=int)  
    return df_encoded

def eda_processing_pipeline(df: pd.DataFrame, var_list: List[str]) -> pd.DataFrame:
    
    df_t1 = transform_ABUSOSUBS(df)
    df_t2 = transform_AUTOEFIC(df_t1)
    df_t3 = transform_CONVIVEN_beta(df_t2)
    df_t4 = transform_IMPULS(df_t3)
    df_t5 = transform_APOYO(df_t4)
    df_t6 = transform_VEXPQUIEN(df_t5)
    df_t7 = transform_VEXPCONTACT(df_t6)
    df_t8 = transform_VMQUIEN(df_t7)
    df_t9 = transform_WQUIEN(df_t8)
    df_t10 = transform_PAIS(df_t9)
    df_t11 = transform_MORAL(df_t10)
    df_t12 = transform_PEXPCONTACT(df_t11)
    df_t13 = transform_VSQUIEN(df_t12)
    df_t14 = transform_VPBULL(df_t13)
    df_t15 = transform_PPBULL(df_t14)
    df_t16 = transform_GENEROBIN_ORIENTSEXBN(df_t15)
    df_t17 = nan_imputer(df_t16)
    df_t18 = get_dummies(df_t17)

    return df_t18[var_list]

def transform_df_to_pca(df_input, scaler_path, means_csv_path, pca_path):
    """
    Transform input DataFrame into PCA component scores.

    This function performs the following steps:
      1. Load a pre-fitted MinMaxScaler, PCA model, and feature means from disk.
      2. Ensure the input DataFrame contains all required features in the correct order.
      3. Apply MinMax scaling to the raw feature values.
      4. Center the scaled data by subtracting the previously computed means.
      5. Project the centered data onto the PCA components.
      6. Return a DataFrame with PCA scores for each input record.

    Parameters
    ----------
    df_input : pd.DataFrame
        DataFrame containing one or more samples.
        Each sample must include the same feature columns used during PCA training.
    scaler_path : str
        Filepath to the serialized MinMaxScaler (.pkl).
    means_csv_path : str
        Filepath to the CSV containing feature means (column 'mean'), indexed by feature name.
    pca_path : str
        Filepath to the serialized PCA model (.pkl).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns 'PC1', 'PC2', ..., corresponding to the projected PCA component values.

    Raises
    ------
    KeyError
        If the input DataFrame is missing any feature columns required by the scaler and PCA.
    """
    # 1) Load artifacts
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    means_df = pd.read_csv(means_csv_path, index_col=0)
    
    if 'mean' in means_df.columns:
        means_arr = means_df['mean'].values
    else:
        means_arr = means_df.squeeze().values

    # 2) Order columns by trained feature names
    feature_names = means_df.index.tolist()
    
    missing = set(feature_names) - set(df_input.columns)
    if missing:
        raise KeyError(f"Missing feature columns in input DataFrame: {missing}")
    
    df_input = df_input[feature_names]

    # 3) Scale features to [0, 1]
    X_scaled = scaler.transform(df_input.values)

    # 4) Center scaled features subtracting training means
    X_centered = X_scaled - means_arr

    # 5) Project onto PCA components
    components = pca.transform(X_centered)

    # 6) Create DataFrame with PCA components
    pc_labels = [f"PC{i+1}" for i in range(components.shape[1])]
    df_pca = pd.DataFrame(components, columns=pc_labels, index=df_input.index)
    
    return df_pca
def classify_dt_pcs_df(df_pca, pruned_clf_path, n_components=18):
    """
    Apply pruned DecisionTreeClassifier to PCA components DataFrame and return predictions.

    Parameters
    ----------
    df_pca : pd.DataFrame
        DataFrame containing PCA components (PC1, PC2, ...) as columns.
    pruned_clf_path : str
        Filepath to the serialized pruned DecisionTreeClassifier (.joblib).
    n_components : int
        Number of PCA components to select for classification (default 18).

    Returns
    -------
    tuple
        (list of predicted labels, list of corresponding probabilities)

    Raises
    ------
    KeyError
        If expected PCA columns are missing in the input.
    """
    # Select first n_components columns
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    
    missing_pcs = set(pc_cols) - set(df_pca.columns)
    if missing_pcs:
        raise KeyError(f"Missing PCA columns: {missing_pcs}")
    
    X = df_pca[pc_cols].values

    # Load pruned classifier
    clf = joblib.load(pruned_clf_path)

    # Predict labels and probabilities
    preds = clf.predict(X)
    probas = clf.predict_proba(X)

    # Extract probability of the predicted class for each sample
    probabilities = []
    for pred, proba_row in zip(preds, probas):
        try:
            class_index = list(clf.classes_).index(pred)
        except ValueError:
            class_index = int(np.argmax(proba_row))
        probabilities.append(float(proba_row[class_index]))

    return list(preds), probabilities
def classify_pcs_nn_df(df_pca, nn_path, n_components=22):
    """
    Classify using a neural network on PCA components DataFrame.

    Parameters
    ----------
    df_pca : pd.DataFrame
        DataFrame containing PCA components (PC1, PC2, ...) as columns.
    nn_path : str
        Filepath to the Keras model.
    n_components : int
        Number of PCA components to use (default 22).

    Returns
    -------
    tuple
        (list of predicted labels, list of corresponding probabilities)

    Raises
    ------
    KeyError
        If expected PCA columns are missing in the input.
    """
    # Select PCs
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    missing = set(pc_cols) - set(df_pca.columns)
    if missing:
        raise KeyError(f"Missing PCA columns: {missing}")
    
    X = df_pca[pc_cols].values

    # Load NN model
    model = load_model(nn_path)

    # Predict probabilities
    probas = model.predict(X)
    probas = np.asarray(probas)

    # Determine predictions and probabilities
    predicted_labels = []
    probabilities = []
    
    if probas.ndim == 2 and probas.shape[1] == 1:
        # Binary classification with single output
        for p in probas.flatten():
            label = int(p >= 0.5)
            prob = float(p) if label == 1 else float(1 - p)
            predicted_labels.append(label)
            probabilities.append(prob)
    else:
        # Multi-class classification
        for p in probas:
            idx = int(np.argmax(p))
            predicted_labels.append(idx)
            probabilities.append(float(p[idx]))

    return predicted_labels, probabilities
def get_predictions(
    df: pd.DataFrame, 
    var_global: List[str], 
    scaler_path=scaler_path, 
    means_csv_path=means_csv_path, 
    pca_path=pca_path,
    pruned_clf_path=pruned_clf_path, 
    nn_path=nn_path,
    dt_pca_components = 18,
    nn_pca_components = 22
) -> pd.DataFrame:
    
    feat_df = df[var_global].drop(columns=["GENERO_BIN_2","ORIENTSEX.BN_3"]).reset_index(drop=True)
    pca_df = transform_df_to_pca(feat_df,scaler_path, means_csv_path, pca_path)
    dt_list_pred, dt_list_prob = classify_dt_pcs_df(pca_df, pruned_clf_path, n_components=18)
    nn_list_pred, nn_list_prob = classify_pcs_nn_df(pca_df, nn_path, n_components=22)
    
    df["VICTIM_pred"] = dt_list_pred
    df["VICTIM_prob"] = dt_list_prob
    df["PERPETRATOR_pred"] = nn_list_pred
    df["PERPETRATOR_prob"] = nn_list_prob
    
    return df


#PREDICTOR QUIZ processor---------------------------------------------------
def get_gender_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
        Given a DataFrame with a column 'GENERO.BN' containing binary values (0 or 1),
        return a new DataFrame with two dummy columns:
        - 'GENERO_BIN_0': 1 where 'GENERO.BN' == 0, else 0
        - 'GENERO_BIN_1': 1 where 'GENERO.BN' == 1, else 0

        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame containing the 'GENERO.BN' column.

        Returns:
        --------
        pd.DataFrame
            DataFrame with the two dummy columns.
    """
    if 'GENERO.BN' not in df.columns:
        raise KeyError("Input DataFrame must contain a 'GENERO.BN' column.")
    
    # Create dummy columns
    gen0 = (df['GENERO.BN'] == 0).astype(int)
    gen1 = (df['GENERO.BN'] == 1).astype(int)
    
    # Build and return the result
    dummies = pd.DataFrame({
        'GENERO_BIN_0': gen0,
        'GENERO_BIN_1': gen1
    }, index=df.index)
    
    return dummies
def get_orientsex_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with a column 'ORIENTSEX.BN' containing binary values (0 or 1),
    return a new DataFrame with two dummy columns:
      - 'ORIENTSEX.BN_1': 1 where 'ORIENTSEX.BN' == 0, else 0
      - 'ORIENTSEX.BN_2': 1 where 'ORIENTSEX.BN' == 1, else 0

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing the 'ORIENTSEX.BN' column.

    Returns:
    --------
    pd.DataFrame
        DataFrame with the two dummy columns.
    """
    if 'ORIENTSEX.BN' not in df.columns:
        raise KeyError("Input DataFrame must contain an 'ORIENTSEX.BN' column.")
    
    # Create dummy columns
    orient0 = (df['ORIENTSEX.BN'] == 0).astype(int)
    orient1 = (df['ORIENTSEX.BN'] == 1).astype(int)
    
    # Build and return the result
    dummies = pd.DataFrame({
        'ORIENTSEX.BN_1': orient1,
        'ORIENTSEX.BN_2': orient0
    }, index=df.index)
    
    return dummies
def get_useful_age(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with a column 'EDAD' containing integer ages (13–18),
    return a new DataFrame with the same column 'EDAD', but with:
      - All values of 13 replaced by 14
      - All values of 18 replaced by 17
      - All other values unchanged

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing the 'EDAD' column with integer values 13 through 18.

    Returns:
    --------
    pd.DataFrame
        DataFrame with the 'EDAD' column after applying the specified replacements.
    """
    if 'EDAD' not in df.columns:
        raise KeyError("Input DataFrame must contain an 'EDAD' column.")
    
    # Define mapping and apply it
    mapping = {13: 14, 18: 17}
    df_converted = df.copy()
    df_converted['EDAD'] = df_converted['EDAD'].replace(mapping)
    
    # Return only the EDAD column as a new DataFrame
    return df_converted[['EDAD']]
def get_useful_conviven(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with integer columns 'CONVIVEN6' and 'CONVIVEN7' (0 or 1),
    returns a new DataFrame with a single column 'CONVIVEN.6' where each value is:
      - 1 if either 'CONVIVEN6' or 'CONVIVEN7' is 1
      - 0 if both are 0
    """
    # Compute logical OR (bitwise) and ensure integer type
    result = pd.DataFrame({
        'CONVIVEN.6': (df['CONVIVEN6'] | df['CONVIVEN7']).astype(int)
    })
    return result

def get_autoefic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with integer columns 'AUTOEFIC1' through 'AUTOEFIC5',
    compute per-row:
      - AUTOEFIC.MEAN: the average of AUTOEFIC1,…,AUTOEFIC5
      - AUTOEFIC.VAR: the population variance of AUTOEFIC1,…,AUTOEFIC5

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing columns 'AUTOEFIC1', 'AUTOEFIC2', ..., 'AUTOEFIC5'.

    Returns:
    --------
    pd.DataFrame
        DataFrame with two columns:
          - 'AUTOEFIC.MEAN'
          - 'AUTOEFIC.VAR'
    """
    required = [f'AUTOEFIC{i}' for i in range(1, 6)]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input DataFrame is missing columns: {missing}")
    
    # Compute mean and variance (population, ddof=0)
    mean_series = df[required].mean(axis=1)
    var_series = df[required].var(axis=1, ddof=0)
    
    return pd.DataFrame({
        'AUTOEFIC.MEAN': mean_series,
        'AUTOEFIC.VAR': var_series
    }, index=df.index)
def get_apoyo_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with integer columns 'APOYO1' through 'APOYO7',
    compute per-row:
      - APOYO.MEAN: the average of APOYO1,…,APOYO7
      - APOYO.VAR: the population variance of APOYO1,…,APOYO7
      - APOYO.MEDIAN: the median of APOYO1,…,APOYO7

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing columns 'APOYO1', 'APOYO2', ..., 'APOYO7'.

    Returns:
    --------
    pd.DataFrame
        DataFrame with three columns:
          - 'APOYO.MEAN'
          - 'APOYO.VAR'
          - 'APOYO.MEDIAN'
    """
    required = [f'APOYO{i}' for i in range(1, 8)]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input DataFrame is missing columns: {missing}")

    # Compute statistics (population variance, ddof=0)
    mean_series = df[required].mean(axis=1)
    var_series = df[required].var(axis=1, ddof=0)
    median_series = df[required].median(axis=1)

    return pd.DataFrame({
        'APOYO.MEAN': mean_series,
        'APOYO.VAR': var_series,
        'APOYO.MEDIAN': median_series
    }, index=df.index)
def get_impuls_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with integer columns 'IMPULS1' through 'IMPULS8',
    compute per-row:
      - IMPULS.MEAN: the average of IMPULS1,…,IMPULS8
      - IMPULS.VAR: the population variance of IMPULS1,…,IMPULS8
      - IMPULS.MEDIAN: the median of IMPULS1,…,IMPULS8

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing columns 'IMPULS1', 'IMPULS2', ..., 'IMPULS8'.

    Returns:
    --------
    pd.DataFrame
        DataFrame with three columns:
          - 'IMPULS.MEAN'
          - 'IMPULS.VAR'
          - 'IMPULS.MEDIAN'
    """
    required = [f'IMPULS{i}' for i in range(1, 9)]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input DataFrame is missing columns: {missing}")

    # Compute statistics (population variance, ddof=0)
    mean_series = df[required].mean(axis=1)
    var_series = df[required].var(axis=1, ddof=0)
    median_series = df[required].median(axis=1)

    return pd.DataFrame({
        'IMPULS.MEAN': mean_series,
        'IMPULS.VAR': var_series,
        'IMPULS.MEDIAN': median_series
    }, index=df.index)
def get_moral_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with integer columns 'MORAL1' through 'MORAL5',
    compute per-row:
      - MORAL.MEAN: the average of MORAL1,…,MORAL5
      - MORAL.VAR: the population variance of MORAL1,…,MORAL5

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing columns 'MORAL1', 'MORAL2', ..., 'MORAL5'.

    Returns:
    --------
    pd.DataFrame
        DataFrame with two columns:
          - 'MORAL.MEAN'
          - 'MORAL.VAR'
    """
    required = [f'MORAL{i}' for i in range(1, 6)]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input DataFrame is missing columns: {missing}")

    # Compute statistics (population variance, ddof=0)
    mean_series = df[required].mean(axis=1)
    var_series = df[required].var(axis=1, ddof=0)

    return pd.DataFrame({
        'MORAL.MEAN': mean_series,
        'MORAL.VAR': var_series
    }, index=df.index)
def get_porno_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with integer columns 'PORNO1' and 'PORNO2',
    compute per-row:
      - PORNO.T: the maximum of PORNO1 and PORNO2

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame containing 'PORNO1' and 'PORNO2'.

    Returns:
    --------
    pd.DataFrame
        DataFrame with a single column 'PORNO.T' containing the per-row maximum.
    """
    required = ['PORNO1', 'PORNO2']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Input DataFrame is missing columns: {missing}")

    # Compute maximum
    max_series = df[required].max(axis=1)

    return pd.DataFrame({'PORNO.T': max_series}, index=df.index)
def get_model_vars(df:pd.DataFrame):
    """
        Extracts and computes a comprehensive set of model-ready features from a raw survey DataFrame.

        This function performs the following transformations:
        1. **One-hot encoding** for binary indicators:
            - `GENERO.BN` → `GENERO_BIN_0`, `GENERO_BIN_1`
            - `ORIENTSEX.BN` → `ORIENTSEX.BN_1`, `ORIENTSEX.BN_2`
        2. **Age normalization** by mapping boundary values:
            - Ages `13` → `14`, `18` → `17`
        3. **Co-residence consolidation**:
            - Combines `CONVIVEN6` and `CONVIVEN7` into `CONVIVEN.6` via logical OR
        4. **Scale statistics** for multi-item measures:
            - `AUTOEFIC1–5`: mean (`AUTOEFIC.MEAN`), variance (`AUTOEFIC.VAR`)
            - `APOYO1–7`: mean (`APOYO.MEAN`), variance (`APOYO.VAR`), median (`APOYO.MEDIAN`)
            - `IMPULS1–8`: mean (`IMPULS.MEAN`), variance (`IMPULS.VAR`), median (`IMPULS.MEDIAN`)
            - `MORAL1–5`: mean (`MORAL.MEAN`), variance (`MORAL.VAR`)
        5. **Item summation** for paired measures:
            - `PORNO1 + PORNO2` → `PORNO.T`
        6. **Concatenation** of derived features with original DataFrame.
        7. **Column renaming** and reordering for consistency.
        8. **Fill missing** values with zeros.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing these required columns:
            - Binary flags: `PAIS.BN`, `ETNIA.BN`, `GENERO.BN`, `ORIENTSEX.BN`, `FUGAS.BN`
            - Age: `EDAD` (integers from 13 to 18)
            - Co-residence: `CONVIVEN1`–`CONVIVEN7`
            - Scale items:
                - `AUTOEFIC1`–`AUTOEFIC5`
                - `APOYO1`–`APOYO7`
                - `IMPULS1`–`IMPULS8`
                - `MORAL1`–`MORAL5`
            - Paired sum items: `PORNO1`, `PORNO2`

        Returns
        -------
        pd.DataFrame
            A new DataFrame (`df_final`) with the following features in this order:

            PAÍS, ETNIA.BN, EDAD, FUGAS.BN, ABUSOSUBS1, ABUSOSUBS2,
            CONVIVEN.1, CONVIVEN.2, CONVIVEN.3, CONVIVEN.4, CONVIVEN.5,
            CONVIVEN.6, AUTOEFIC.MEAN, AUTOEFIC.VAR, IMPULS.MEAN,
            IMPULS.MEDIAN, IMPULS.VAR, APOYO.MEAN, APOYO.MEDIAN,
            APOYO.VAR, MORAL.MEAN, MORAL.VAR, PORNO.T,
            GENERO_BIN_0, GENERO_BIN_1, ORIENTSEX.BN_1, ORIENTSEX.BN_2

            All missing values in these columns are filled with zero.

        Raises
        ------
        KeyError
            If any of the required columns are missing from the input DataFrame.

        Example
        -------
        >>> import pandas as pd
        >>> from data_processor import get_model_vars
        >>> df_raw = pd.read_excel("survey.xlsx")
        >>> df_features = get_model_vars(df_raw)
        >>> df_features.head()
    """
    df_gender = get_gender_dummies(df)
    df_orient = get_orientsex_dummies(df)
    df_age = get_useful_age(df)
    df_conv = get_useful_conviven(df)
    df_autoefic = get_autoefic_statistics(df)
    df_apoyo = get_apoyo_statistics(df)
    df_impuls = get_impuls_statistics(df)
    df_moral = get_moral_statistics(df)
    df_porno = get_porno_total(df)
    
    df_combined = pd.concat([
        df,
        df_gender,
        df_orient,
        df_age,
        df_conv,
        df_autoefic,
        df_apoyo,
        df_impuls,
        df_moral,
        df_porno
    ], axis=1)
    
    df_combined.rename(
        columns={
            'PAIS.BN':'PAÍS',
            'CONVIVEN1':'CONVIVEN.1',
            'CONVIVEN2':'CONVIVEN.2',
            'CONVIVEN3':'CONVIVEN.3',
            'CONVIVEN4':'CONVIVEN.4',
            'CONVIVEN5':'CONVIVEN.5',
        },
        inplace=True
    )

    
    column_order = ['PAÍS', 'ETNIA.BN', 'EDAD', 'FUGAS.BN', 'ABUSOSUBS1', 'ABUSOSUBS2',
       'CONVIVEN.1', 'CONVIVEN.2', 'CONVIVEN.3', 'CONVIVEN.4', 'CONVIVEN.5',
       'CONVIVEN.6', 'AUTOEFIC.MEAN', 'AUTOEFIC.VAR', 'IMPULS.MEAN',
       'IMPULS.MEDIAN', 'IMPULS.VAR', 'APOYO.MEAN', 'APOYO.MEDIAN',
       'APOYO.VAR', 'MORAL.MEAN', 'MORAL.VAR', 'PORNO.T', 'GENERO_BIN_0',
       'GENERO_BIN_1', 'ORIENTSEX.BN_1', 'ORIENTSEX.BN_2'
    ]
    
    df_final = df_combined[column_order]
    df_final = df_final.fillna(0)
    
    return df_final
def transform_json_to_pca(json_input, scaler_path=scaler_path, means_csv_path=means_csv_path, pca_path=pca_path):
    """
    Transform input JSON records into PCA component scores.

    This function performs the following steps:
      1. Parse the input JSON string or Python object into a list of feature dictionaries.
      2. Load a pre-fitted MinMaxScaler, PCA model, and feature means from disk.
      3. Ensure the input data contains all required features in the correct order.
      4. Apply MinMax scaling to the raw feature values.
      5. Center the scaled data by subtracting the previously computed means.
      6. Project the centered data onto the PCA components.
      7. Return a JSON string with PCA scores for each input record.

    Parameters
    ----------
    json_input : str | dict | list of dict
        JSON representation (string, dict, or list) of one or more samples.
        Each sample must include the same feature keys used during PCA training.
    scaler_path : str
        Filepath to the serialized MinMaxScaler (.pkl).
    means_csv_path : str
        Filepath to the CSV containing feature means (column 'mean'), indexed by feature name.
    pca_path : str
        Filepath to the serialized PCA model (.pkl).

    Returns
    -------
    str
        JSON-formatted string representing a list of objects. Each object contains
        keys 'PC1', 'PC2', ..., corresponding to the projected PCA component values.

    Raises
    ------
    KeyError
        If the input JSON is missing any feature columns required by the scaler and PCA.
    ValueError
        If the input cannot be parsed or is of incorrect type.
    """
    # 1) Parse input JSON into Python objects
    if isinstance(json_input, str):
        try:
            data = json.loads(json_input)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON input: {e}")
    else:
        data = json_input

    # Normalize to list of dicts
    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        raise ValueError("Input must be a JSON string, dict, or list of dicts.")

    # 2) Load artifacts
    scaler = joblib.load(scaler_path)
    pca = joblib.load(pca_path)
    means_df = pd.read_csv(means_csv_path, index_col=0)
    if 'mean' in means_df.columns:
        means_arr = means_df['mean'].values
    else:
        means_arr = means_df.squeeze().values

    # 3) Build DataFrame from input, ordering columns by trained feature names
    feature_names = means_df.index.tolist()
    df_input = pd.DataFrame(data)

    missing = set(feature_names) - set(df_input.columns)
    if missing:
        raise KeyError(f"Missing feature columns in input JSON: {missing}")
    df_input = df_input[feature_names]

    # 4) Scale features to [0, 1]
    X_scaled = scaler.transform(df_input.values)

    # 5) Center scaled features subtracting training means
    X_centered = X_scaled - means_arr

    # 6) Project onto PCA components
    components = pca.transform(X_centered)

    # 7) Format output JSON with PCA scores
    pc_labels = [f"PC{i+1}" for i in range(components.shape[1])]
    results = [dict(zip(pc_labels, row)) for row in components]

    return json.dumps(results, ensure_ascii=False)
def classify_dt_pcs(json_pca,pruned_clf_path=pruned_clf_path,n_components=18):
    """
    Load PCA scores from JSON, select the first n_components,
    apply the pruned DecisionTreeClassifier, and return predictions.

    Parameters
    ----------
    json_pca : str | dict | list of dict
        JSON string or Python object containing PCA scores for each sample,
        with keys 'PC1', 'PC2', ..., up to the total number of components.
    pruned_clf_path : str
        Filepath to the serialized pruned DecisionTreeClassifier (.joblib).
    n_components : int
        Number of PCA components to select for classification (default 18).

    Returns
    -------
    str
        JSON-formatted string representing a list of objects. Each object contains:
          - 'predicted_label': int (class predicted by the DecisionTreeClassifier)
          - 'probability': float (probability of the predicted class)

    Raises
    ------
    ValueError
        If input JSON cannot be parsed or has insufficient PCA components.
    KeyError
        If expected PCA keys are missing in the input.
    """
    # 1) Parse input JSON
    if isinstance(json_pca, str):
        try:
            pca_data = json.loads(json_pca)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON for PCA data: {e}")
    else:
        pca_data = json_pca

    if isinstance(pca_data, dict):
        pca_data = [pca_data]
    elif not isinstance(pca_data, list):
        raise ValueError("PCA input must be a JSON string, dict, or list of dicts.")

    # 2) Build DataFrame and select first n_components columns
    df_pca = pd.DataFrame(pca_data)
    pc_cols = [f"PC{i+1}" for i in range(n_components)]

    missing_pcs = set(pc_cols) - set(df_pca.columns)
    if missing_pcs:
        raise KeyError(f"Missing PCA columns: {missing_pcs}")
    X = df_pca[pc_cols].values

    # 3) Load pruned classifier
    clf = joblib.load(pruned_clf_path)

    # 4) Predict labels and probabilities
    preds = clf.predict(X)
    probas = clf.predict_proba(X)

    # 5) Extract probability of the predicted class for each sample
    pred_results = []
    for pred, proba_row in zip(preds, probas):
        # Find the class index corresponding to the predicted label
        try:
            class_index = list(clf.classes_).index(pred)
        except ValueError:
            # If predicted label not in classes_, default to highest probability
            class_index = int(np.argmax(proba_row))
        prob = float(proba_row[class_index])
        pred_results.append({
            'predicted_label': int(pred),
            'probability': prob
        })

    return json.dumps(pred_results, ensure_ascii=False)
def classify_pcs_nn(json_pca,nn_path=nn_path,n_components=22):
    """
    Classify using a neural network on first n_components PCA scores.

    Steps:
      1. Parse JSON of PCA scores.
      2. Build DataFrame and select PC1..PCn columns.
      3. Load Keras model and predict.
      4. Return JSON list of {'predicted_label','probability'}.

    Raises
    ------
    KeyError, ValueError
    """
    # Parse PCA JSON
    data = json.loads(json_pca) if isinstance(json_pca, str) else json_pca
    data = [data] if isinstance(data, dict) else data
    df_pca = pd.DataFrame(data)

    # Select PCs
    pc_cols = [f"PC{i+1}" for i in range(n_components)]
    missing = set(pc_cols) - set(df_pca.columns)
    if missing:
        raise KeyError(f"Missing PCA columns: {missing}")
    X = df_pca[pc_cols].values

    # Load NN model
    model = load_model(nn_path)

    # Predict probabilities
    probas = model.predict(X)
    probas = np.asarray(probas)

    # Determine predictions
    results = []
    if probas.ndim == 2 and probas.shape[1] == 1:
        for p in probas.flatten():
            label = int(p >= 0.5)
            prob = float(p) if label == 1 else float(1 - p)
            results.append({'predicted_label': label, 'probability': prob})
    else:
        for p in probas:
            idx = int(np.argmax(p))
            results.append({'predicted_label': idx, 'probability': float(p[idx])})

    return json.dumps(results, ensure_ascii=False)

def make_pc_frame(pcs_full: dict, upto: int) -> pd.DataFrame:
    cols = [f"PC{i}" for i in range(1, upto + 1)]
    missing = [c for c in cols if c not in pcs_full]
    if missing:
        raise KeyError(f"Missing PCA keys: {missing}")
    return pd.DataFrame([[pcs_full[c] for c in cols]], columns=cols)
def infer_pc_shap_dt(
    pcs_full: dict,
    pruned_clf_path: str = pruned_clf_path,
    model_output: str = "raw",   # use "raw" or "probability" (with background)
) -> dict:
    df_pca = make_pc_frame(pcs_full, 18)
    clf = joblib.load(pruned_clf_path)

    # Choose explainer
    if model_output == "probability":
        background_df = pd.DataFrame(np.zeros((50, df_pca.shape[1])), columns=df_pca.columns)
        explainer = shap.TreeExplainer(
            clf,
            data=background_df,
            model_output="probability",
            feature_perturbation="interventional",
        )
    else:
        explainer = shap.TreeExplainer(clf, model_output="raw")

    # Predicted class index
    pred_label = clf.predict(df_pca.values)[0]
    class_idx = list(clf.classes_).index(pred_label)

    shap_values = explainer.shap_values(df_pca)

    # --- Robustly pick the correct 1D vector of length n_features ---
    if isinstance(shap_values, list):
        # Typical older API: list per class -> pick predicted class, first (and only) sample
        sv = np.asarray(shap_values[class_idx][0])             # (n_features,)
    else:
        arr = np.asarray(shap_values)
        if arr.ndim == 3:
            # (n_samples, n_features, n_classes)
            sv = arr[0, :, class_idx]                          # (n_features,)
        elif arr.ndim == 2:
            # (n_samples, n_features)
            sv = arr[0]                                        # (n_features,)
        elif arr.ndim == 1 and arr.size == df_pca.shape[1]:
            # Already (n_features,)
            sv = arr
        else:
            raise ValueError(f"Unexpected SHAP shape {arr.shape}; can’t align to features.")

    sv = np.asarray(sv).astype(float)
    if sv.size != df_pca.shape[1]:
        raise ValueError(f"Unexpected SHAP shape {sv.shape}; expected {df_pca.shape[1]} values.")

    return dict(zip(df_pca.columns, sv.tolist()))
def infer_pc_shap_nn(pcs_full: dict,nn_path: str = nn_path,background: list | None = None,nsamples: int | str = "auto") -> dict:

    df_pca = make_pc_frame(pcs_full, 22)

    # Background: use your own representative PC rows if you have them
    if background is None:
        background_df = pd.DataFrame([{c: 0.0 for c in df_pca.columns}])
    else:
        bg = pd.DataFrame(background)
        background_df = bg[df_pca.columns]  # align & select

    model = load_model(nn_path)

    # Sanity: NN must expect 22 features
    in_dim = model.input_shape[-1]
    assert in_dim == df_pca.shape[1], f"NN expects {in_dim} features, got {df_pca.shape[1]}"

    f = lambda X: model.predict(X, verbose=0)

    explainer = shap.KernelExplainer(f, background_df.values)
    shap_values = explainer.shap_values(df_pca.values, nsamples=nsamples)

    probs = f(df_pca.values)  # (1, C) or (1,1)
    if probs.ndim == 2 and probs.shape[1] == 1:
        # Binary single-output sigmoid
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) == 2 else shap_values[0][0]
        else:
            sv = shap_values[0]
    else:
        # Multiclass: take predicted class
        class_idx = int(np.argmax(probs[0]))
        sv = shap_values[class_idx][0] if isinstance(shap_values, list) else shap_values[0]

    return dict(zip(df_pca.columns, map(float, sv)))
def shap_original_from_pcs(feat_pc_df: pd.DataFrame,pc_shap: Dict[str, float],feature_col: str = "feature") -> Dict[str, float]:
    """
    Given a dataframe of feature→PC importances and a dict of PC→SHAP weights,
    compute the SHAP value of each original feature as:
        shap_orig(feature) = sum_i [ importance(feature, PCi) * shap(PCi) ]

    Parameters
    ----------
    feat_pc_df : pd.DataFrame
        Must contain one row per original feature, a column named `feature_col`,
        and columns for each PC (e.g. "PC1", "PC2", …).
    pc_shap : dict[str, float]
        Mapping from PC name (must match columns in feat_pc_df) to its SHAP weight.
    feature_col : str, default="feature"
        Name of the column in feat_pc_df that holds the original feature names.

    Returns
    -------
    dict[str, float]
        Mapping from each original feature to its reconstructed SHAP value.
    """
    # — Ensure all PCs in the dict exist in the DataFrame
    missing = set(pc_shap) - set(feat_pc_df.columns)
    if missing:
        raise KeyError(f"PC columns not found in DataFrame: {missing}")

    # — Multiply each PC‐column by its SHAP weight
    #   and sum across PCs to get each feature's overall SHAP
    pcs = list(pc_shap.keys())
    weighted = feat_pc_df[pcs].multiply(
        pd.Series(pc_shap), axis=1
    )
    feat_pc_df = feat_pc_df.copy()
    feat_pc_df["shap_original"] = weighted.sum(axis=1)

    # — Build and return final dict
    return dict(zip(feat_pc_df[feature_col], feat_pc_df["shap_original"]))
def shap_to_percent_with_sign(shap_vals: Dict[str, float]) -> Dict[str, Tuple[float, int]]:
    """
    Given a dict of SHAP values per original feature (which may be negative),
    return a dict mapping each feature to a tuple:
      (percentage_of_total_importance, sign)

    - percentage_of_total_importance is computed as
        abs(shap_val) / sum(abs(all shap_vals)) * 100
      so that you see each feature’s share of the total importance.
    - sign is +1 if the original SHAP ≥ 0, else -1.

    If all SHAP values are zero, every percentage will be 0.0.

    Parameters
    ----------
    shap_vals : dict[str, float]
        Original SHAP values by feature.

    Returns
    -------
    dict[str, (float, int)]
        feature → (percent_importance, sign_flag)
    """
    if not shap_vals:
        return {}

    # Compute absolute values and their total
    abs_vals = {feat: abs(val) for feat, val in shap_vals.items()}
    total = sum(abs_vals.values())

    # Avoid division by zero if all values are zero
    if total == 0:
        return {feat: (0.0, 1 if shap_vals[feat] >= 0 else -1)
                for feat in shap_vals}

    # Build the output dict
    result: Dict[str, Tuple[float, int]] = {}
    for feat, val in shap_vals.items():
        pct = abs_vals[feat] / total * 100
        sign = 1 if val >= 0 else -1
        result[feat] = (round(pct,3), sign)

    return result
def combine_shap_percent(shap_dt_vars, shap_nn_vars):
    """
    Combine two SHAP‐percent dicts into one, per-field, by:
      - If both signs are the same: sum the magnitudes and keep that sign.
      - If signs differ: take the absolute difference of magnitudes and adopt 
        the sign of the larger-magnitude entry.
    
    Parameters
    ----------
    shap_dt_vars : dict[str, tuple[float, int]]
        e.g. { "PAÍS": (3.89, -1), ... }
    shap_nn_vars : dict[str, tuple[float, int]]
        same format as shap_dt_vars
        
    Returns
    -------
    dict[str, tuple[float, int]]
        Combined dict mapping each key to (new_magnitude, new_sign).
    """
    combined = {}
    for key, (v_dt, s_dt) in shap_dt_vars.items():
        if key not in shap_nn_vars:
            raise KeyError(f"Key {key!r} missing from second dict")
        v_nn, s_nn = shap_nn_vars[key]

        if s_dt == s_nn:
            # same sign → add magnitudes
            combined_val = v_dt + v_nn
            combined_sign = s_dt
        else:
            # different signs → difference, keep sign of larger magnitude
            combined_val = abs(v_dt - v_nn)
            combined_sign = s_dt if v_dt >= v_nn else s_nn

        combined[key] = (combined_val, combined_sign)
    return combined
def rebuild_shap_dict(float_dict, original_shap_dict):
    """
    Given:
      float_dict          : dict[str, float]
      original_shap_dict  : dict[str, tuple[float, int]]
                            (the one you originally filtered from,
                             so it still contains the signs)
    Returns:
      dict[str, tuple[float, int]]
        For each key, ( float_dict[key], original_shap_dict[key][1] )
    """
    return {
        key: (float_dict[key][0], original_shap_dict[key][1])
        for key in float_dict
    }

def actualize_country(dict_vars: Dict[str, Any], country: str) -> Dict[str, Any]:
    if country != "--":
        dict_vars["PAÍS"] = 1 if country == "SPAIN" else 0
    return dict_vars
def actualize_ethnic(dict_vars: Dict[str, Any], european_list: List[str], ethnic: str) -> Dict[str, Any]:
    if ethnic != "--":
        dict_vars["ETNIA.BN"] = 1 if ethnic in european_list else 0
    return dict_vars
def actualize_sex(dict_vars: Dict[str, Any], sex: str) -> Dict[str, Any]:
    if sex != "--":
        dict_vars["GENERO_BIN_0"] = 1 if sex == "FAMALE" else 0
        dict_vars["GENERO_BIN_1"] = 1 if sex == "MALE" else 0
    return dict_vars
def actualize_orientation(dict_vars: Dict[str, Any], orientation: str) -> Dict[str, Any]:
    if orientation != "--":
        dict_vars["ORIENTSEX.BN_1"] = 1 if orientation == "HETEROSEXUAL" else 0
        dict_vars["ORIENTSEX.BN_2"] = 1 if orientation != "HETEROSEXUAL" else 0
    return dict_vars
def actualize_age(dict_vars: Dict[str, Any], age: int) -> Dict[str, Any]:
    """
    Asigna EDAD transformando 13->14 y 18->17, en otros casos usa el valor dado.
    """
    if not isinstance(age, int):
        return dict_vars
    if age == 13:
        dict_vars["EDAD"] = 14
    elif age == 18:
        dict_vars["EDAD"] = 17
    else:
        dict_vars["EDAD"] = age
    return dict_vars
def actualize_family(dict_vars: Dict[str, Any], family: List[str]) -> Dict[str, Any]:
    mapping = {
        "Mother": "CONVIVEN.1",
        "Father": "CONVIVEN.2",
        "Mother's Partner": "CONVIVEN.3",
        "Father's Partner": "CONVIVEN.4",
        "Siblings/Stepsiblings": "CONVIVEN.5",
        "Aunt/Oncle": "CONVIVEN.6",
        "Grandparents": "CONVIVEN.6",
        "Center": "CONVIVEN.6",
    }
    for member, key in mapping.items():
        if member in family:
            dict_vars[key] = 1
    return dict_vars
def actualize_escape(dict_vars: Dict[str, Any], escape: str) -> Dict[str, Any]:
    if escape in ("Yes", "No"):
        dict_vars["FUGAS.BN"] = 1 if escape == "Yes" else 0
    return dict_vars
def actualize_abusub1(dict_vars: Dict[str, Any], abus1: str) -> Dict[str, Any]:
    mapping = {
        "Never": 0,
        "A few times a year": 2,
        "Monthly": 3,
        "Weekly": 4,
        "Daily or almost daily": 5,
    }
    dict_vars["ABUSOSUBS1"] = mapping.get(abus1, dict_vars.get("ABUSOSUBS1"))
    return dict_vars
def actualize_abusub2(dict_vars: Dict[str, Any], abus2: str) -> Dict[str, Any]:
    mapping = {
        "Never": 0,
        "A few times a year": 2,
        "Monthly": 3,
        "Weekly": 4,
        "Daily or almost daily": 5,
    }
    dict_vars["ABUSOSUBS2"] = mapping.get(abus2, dict_vars.get("ABUSOSUBS2"))
    return dict_vars
def actualize_porn(dict_vars: Dict[str, Any], porn1: str, porn2: str) -> Dict[str, Any]:
    map1 = {"Never": 1, "A few times a year": 2, "Monthly": 3, "Weekly": 4, "Daily or almost daily": 5}
    map2 = {"Never": 1, "A few times a year": 2, "Monthly": 3, "Weekly": 4, "Daily or almost daily": 5}
    val1 = map1.get(porn1, 0)
    val2 = map2.get(porn2, 0)
    dict_vars["PORNO.T"] = max(val1, val2)
    return dict_vars
def actualize_autoefic(dic_vars: Dict[str, Any], *answers: str) -> Dict[str, Any]:
    mapping = {"False": 1, "More false than true": 2, "More true than false": 3, "True": 4}
    valores = [mapping.get(ans, 0) for ans in answers]
    dic_vars["AUTOEFIC.MEAN"] = statistics.mean(valores)
    dic_vars["AUTOEFIC.VAR"] = statistics.pvariance(valores)
    return dic_vars
def actualize_impuls(dic_vars: Dict[str, Any], *answers: str) -> Dict[str, Any]:
    mapping = {
        "Rarely or never":          [4,1,1,4,4,4,1,1],
        "Occasionally":             [3,2,2,3,3,3,2,2],
        "Often":                    [2,3,3,2,2,2,3,3],
        "Always or almost always":  [1,4,4,1,1,1,4,4]
    }
    # Para cada ítem, extraemos el valor según el índice en la lista de mapping
    valores = [mapping.get(ans, [0]*8)[i] for i, ans in enumerate(answers)]
    dic_vars["IMPULS.MEAN"] = statistics.mean(valores)
    dic_vars["IMPULS.VAR"] = statistics.pvariance(valores)
    dic_vars["IMPULS.MEDIAN"] = statistics.median(valores)
    return dic_vars
def actualize_apoyo(dic_vars: Dict[str, Any], *answers: str) -> Dict[str, Any]:
    mapping = {"False":1, "More false than true":2, "More true than false":3, "True":4}
    valores = [mapping.get(ans, 0) for ans in answers]
    dic_vars["APOYO.MEAN"] = statistics.mean(valores)
    dic_vars["APOYO.VAR"] = statistics.pvariance(valores)
    dic_vars["APOYO.MEDIAN"] = statistics.median(valores)
    return dic_vars
def actualize_moral(dic_vars: Dict[str, Any], *answers: str) -> Dict[str, Any]:
    mapping = {"Strongly disagree":1, "Somewhat disagree":2, "Neutral":3, "Somewhat agree":4, "Strongly agree":5}
    valores = [mapping.get(ans, 0) for ans in answers]
    dic_vars["MORAL.MEAN"] = statistics.mean(valores)
    dic_vars["MORAL.VAR"] = statistics.pvariance(valores)
    return dic_vars

def update_dict_vars(
    dict_vars: Dict[str, Any],
    country: str,
    european_list: List[str],
    ethnic: str,
    sex: str,
    orientation: str,
    age: int,
    family: List[str],
    escape: str,
    abus1: str,
    abus2: str,
    porn1: str,
    porn2: str,
    autoefic1: str,
    autoefic2: str,
    autoefic3: str,
    autoefic4: str,
    autoefic5: str,
    impuls1: str,
    impuls2: str,
    impuls3: str,
    impuls4: str,
    impuls5: str,
    impuls6: str,
    impuls7: str,
    impuls8: str,
    apoyo1: str,
    apoyo2: str,
    apoyo3: str,
    apoyo4: str,
    apoyo5: str,
    apoyo6: str,
    apoyo7: str,
    moral1: str,
    moral2: str,
    moral3: str,
    moral4: str,
    moral5: str
) -> Dict[str, Any]:
    calls = [
        (actualize_country, (dict_vars, country)),
        (actualize_ethnic, (dict_vars, european_list, ethnic)),
        (actualize_sex, (dict_vars, sex)),
        (actualize_orientation, (dict_vars, orientation)),
        (actualize_age, (dict_vars, age)),
        (actualize_family, (dict_vars, family)),
        (actualize_escape, (dict_vars, escape)),
        (actualize_abusub1, (dict_vars, abus1)),
        (actualize_abusub2, (dict_vars, abus2)),
        (actualize_porn, (dict_vars, porn1, porn2)),
        (actualize_autoefic, (dict_vars, autoefic1, autoefic2, autoefic3, autoefic4, autoefic5)),
        (actualize_impuls, (dict_vars, impuls1, impuls2, impuls3, impuls4, impuls5, impuls6, impuls7, impuls8)),
        (actualize_apoyo, (dict_vars, apoyo1, apoyo2, apoyo3, apoyo4, apoyo5, apoyo6, apoyo7)),
        (actualize_moral, (dict_vars, moral1, moral2, moral3, moral4, moral5)),
    ]

    for func, args in calls:
        try:
            dict_vars = func(*args)
        except Exception:
            continue
    return dict_vars
def get_formated_message(dict_vars: Dict[str, Any]) -> Union[str, Dict[str, Any]]:
    """
    Genera un mensaje formateado en markdown con los valores no numéricos,
    revisa si todos los campos CONVIVEN son cero y ajusta el mensaje según el caso.
    """
    # 1. Detectar valores no numéricos
    non_numeric = [v for v in dict_vars.values() if not isinstance(v, (int, float))]
    unique_non_numeric = list(set(non_numeric))

    # 2. Obtener campos CONVIVEN
    conv_keys = [k for k in dict_vars.keys() if k.startswith("CONVIVEN.")]
    conv_vals = [dict_vars[k] for k in conv_keys if isinstance(dict_vars[k], (int, float))]
    all_conv_zero = all(v == 0 for v in conv_vals) if conv_vals else True

    # Caso A: hay valores no numéricos
    if unique_non_numeric:
        bullets = "\n".join(f"- {item}" for item in unique_non_numeric)
        message = f"Campos con valores no numéricos:\n\n{bullets}"
        if all_conv_zero:
            message += "\n\nCheck the Habits section and select which persons do you are currently living."
        return message

    # Caso B: todo es numérico pero todos CONVIVEN a 0
    if all_conv_zero:
        return "Check the Habits section and select which persons do you are currently living."

    # Caso C: todo numérico y al menos un CONVIVEN != 0
    return dict_vars

#APP FUNCTIONALITIES-------------------------------------------------------------
def missing_summary(df, colname):
    """
    Returns a Markdown table (as a single plain string built without f-strings)
    showing the count and percentage of missing values in the specified column.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    colname : str
        Name of the column to inspect.

    Returns
    -------
    str
        A single string containing the Markdown table.

    Raises
    ------
    KeyError
        If colname is not in df.
    """
    if colname not in df.columns:
        raise KeyError("Column '{}' not found in DataFrame.".format(colname))
    
    total = len(df)
    missing = df[colname].isna().sum()
    if total:
        pct = missing / total * 100
    else:
        pct = 0.0

    md = (
        "| Column | Missing Count | Missing % |\n"
        "|:-------|--------------:|----------:|\n"
        "| {0} | {1} | {2:.2f}% |"
    ).format(colname, missing, pct)

    return md
def get_unique_values(df: pd.DataFrame, column_name: str) -> list:
    """
    Returns the unique values from a specified column in a DataFrame as a sorted list:
    - If the column is numeric, values are sorted in descending order.
    - Otherwise (e.g. strings), values are sorted alphabetically (A→Z).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame.
    column_name : str
        The name of the column to extract unique values from.

    Returns
    -------
    list
        The unique values from the specified column, sorted as described.

    Raises
    ------
    KeyError
        If the column does not exist in the DataFrame.
    """
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame.")
    
    # Get unique values (preserves NaN if present)
    uniques = df[column_name].unique().tolist()
    
    # Separate out NaNs so they end up at the end of the sorted list
    non_nans = [v for v in uniques if not (isinstance(v, float) and math.isnan(v))]
    nans = [v for v in uniques if (isinstance(v, float) and math.isnan(v))]
    
    # Sort according to dtype
    if is_numeric_dtype(df[column_name]):
        non_nans.sort(reverse=False)
    else:
        non_nans.sort(key=lambda x: str(x))
    
    # Re-attach any NaNs at the end
    return non_nans + nans
def get_range_list(options: list, start, end) -> list:
    """
    Given an ordered list of discrete values and two endpoints,
    returns the sublist of all values between (and including) start and end.

    Parameters
    ----------
    options : list
        The ordered list of discrete values.
    start : any
        The first selected value; must be in options.
    end : any
        The second selected value; must be in options.

    Returns
    -------
    list
        The slice of options from start to end, inclusive.

    Raises
    ------
    ValueError
        If start or end is not found in options.
    """
    if start not in options:
        raise ValueError("Start value {!r} not found in options".format(start))
    if end not in options:
        raise ValueError("End value {!r} not found in options".format(end))

    i_start = options.index(start)
    i_end = options.index(end)

    # Ensure we slice in the correct order
    if i_start <= i_end:
        return options[i_start : i_end + 1]
    else:
        return options[i_end : i_start + 1]
def filter_by_values(df, colname, values):
    """
    Returns a filtered DataFrame containing only the rows where `colname`
    is one of the specified `values`.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame.
    colname : str
        Name of the column to filter on.
    values : list
        List of values to keep in `colname`.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame containing only rows where df[colname] is in `values`.

    Raises
    ------
    KeyError
        If `colname` is not in `df`.
    """
    if colname not in df.columns:
        raise KeyError("Column '{}' not found in DataFrame.".format(colname))
    
    # Use .isin() to filter, and return a copy to avoid SettingWithCopyWarning
    filtered_df = df[df[colname].isin(values)].copy()
    return filtered_df
def json2md_variables_formater(template, data):
    """
    Generate a Markdown report using a custom template and data dictionary.

    Args:
        template (str): A format string with '{description}' and '{value}'.
        data (dict): Mapping of variable names to their assigned values.

    Returns:
        str: Concatenated Markdown sections for each variable.
    """
    sections = []
    for var, val in data.items():
        # Fetch English description or fallback
        desc = taxonomic_dict.get(var, (None, f"#### {var}\nDescription unavailable."))[1]
        sections.append(template.format(description=desc, value=val))
    return "\n\n".join(sections)
def metric_parser(metric: Dict[str, float]) -> Tuple[str, str]:
    """
    Parse a metric dict to determine delta_color and formatted probability.

    Args:
        metric: A dict with keys:
            - "predicted_label": 0 or 1
            - "probability": float

    Returns:
        A tuple (delta_color, value) where:
            - delta_color is "inverse" if predicted_label == 1, else "normal"
            - value is the probability rounded to 2 decimal places with a percent sign
    """
    label = metric.get("predicted_label", 0)
    prob = metric.get("probability", 0.0)

    # Streamlit expects "inverse", not "invers"
    delta_color = "inverse" if label == 1 else "normal"
    label = "High" if label == 1 else "Low"
    value = f"{round(prob, 2)} %"

    return delta_color, label, value
def overlap_metric_parser(metric2: Dict[str, float], metric1: Dict[str, float]) -> Tuple[str, str, str]:
    """
    Parse two metric dicts to determine delta_color, label, and formatted probability.

    Args:
        metric1, metric2: Dicts with keys:
            - "predicted_label": 0 or 1
            - "probability": float

    Returns:
        A tuple (delta_color, label, value) where:
            - delta_color is "inverse" if both predicted_label == 1, else "normal"
            - label is "High" if both predicted_label == 1, else "Low"
            - value is the averaged probability if both == 1, else
              the probability from the metric whose label == 0, formatted as percent
    """
    label1 = metric1.get("predicted_label", 0)
    prob1 = metric1.get("probability", 0.0)
    
    label2 = metric2.get("predicted_label", 0)
    prob2 = metric2.get("probability", 0.0)

    # Both labels = 1 → high / inverse
    if label1 == 1 and label2 == 1:
        delta_color = "inverse"
        label = "High"
        value = f"{round((prob1 + prob2) * 0.5, 2)} %"
    else:
        delta_color = "normal"
        label = "Low"
        # pick the probability of the metric where label == 0
        if label1 == 0:
            value = f"{round(prob1, 2)} %"
        else:
            value = f"{round(prob2, 2)} %"

    return delta_color, label, value



