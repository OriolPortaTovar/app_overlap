#DATA SPlIT
lista_global_vars = [
    'PAÍS', # Binaria muy desbalanceada. [(1,3654),(2,370)]
    'ETNIA.BN', # Binaria muy desbalanceada. [(0,3195),(1,829)]
    'EDAD', # Numerica discreta
    'FUGAS.BN', # Binaria muy desbalanceada. [(0,3462),(1,562)]
    'ABUSOSUBS1', # Numérica discreta muy desbalanceada. [(0,1690),(1,10),(2,1243),(3,691),(4,362),(5,28)]
    'ABUSOSUBS2', # Numérica discreta muy desbalanceada. [(0,801),(1,1632),(2,915),(3,437),(4,209),(5,30)]
    'CONVIVEN.1', # Binaria
    'CONVIVEN.2', # Binaria
    'CONVIVEN.3', # Binaria
    'CONVIVEN.4', # Binaria
    'CONVIVEN.5', # Binaria
    'CONVIVEN.6', # Binaria
    'AUTOEFIC.MEAN', # Numerica Continua
    'AUTOEFIC.VAR', # Numerica Continua
    'IMPULS.MEAN', # Numerica Continua
    'IMPULS.MEDIAN', # Numerica Continua
    'IMPULS.VAR', # Numerica Continua
    'APOYO.MEAN', # Numerica Continua
    'APOYO.MEDIAN', # Numerica Continua
    'APOYO.VAR', # Numerica Continua
    'MORAL.MEAN', # Numerica Continua
    'MORAL.VAR', # Numerica Continua
    'PORNO.T', # Numerica discreta
    'GENERO_BIN_0', # Binaria
    'GENERO_BIN_1', # Binaria
    'GENERO_BIN_2', # Binaria muy desbalanceada. [(0,3919),(1,105)]
    'ORIENTSEX.BN_1', # Binaria muy desbalanceada. [(0,730),(1,3294)]
    'ORIENTSEX.BN_2', # Binaria muy desbalanceada. [(0,3465),(1,559)]
    'ORIENTSEX.BN_3', # Binaria muy desbalanceada. [(0,3853),(1,171)]
]
lista_perpetrador = [
    'PINT.SUM', # Numérica discreta muy desbalanceada. [(0,3844),(1,145),(2,20),(3,10),(4,5)]
    'PEXP.SUM', # Numérica discreta muy desbalanceada. [(0,3959),(1,44),(2,18),(3,3)]
    'PDV.SUM', # Numérica discreta muy desbalanceada. [(0,3823),(1,165),(2,30),(3,6)]
    'PM.SUM', # Numérica discreta muy desbalanceada. [(0,3586),(1,387),(2,51)]
    'PS.SUM', # Numérica discreta muy desbalanceada. [(0,3949),(1,46),(2,10),(3,11),(4,4),(5,4)]
    'PS.ELECT.SUM', # Numérica discreta muy desbalanceada. [(0,3959),(1,53),(2,12)]
    'PS.FÍSICA.SUM', # Numérica discreta muy desbalanceada. [(0,3940),(1,54),(2,7),(3,8),(4,9),(5,4),(6,2)]
    'PC.SUM', # Numérica discreta muy desbalanceada. [(0,3798),(1,176),(2,37),(3,13)]
    'PP.SUM', # Numérica discreta muy desbalanceada. [(0,3632),(1,324),(2,68)]
    'PP.BULL',  # Binaria muy desbalanceada. [(0,3963),(1,61)]
    'PEXP.CONTACT_0', # Binaria muy desbalanceada. [(0,3959),(1,44)]
    'PEXP.CONTACT_1', # Binaria muy desbalanceada. [(0,4011),(1,13)]
    'PEXP.CONTACT_2', # Binaria muy desbalanceada. [(0,4018),(1,6)]
    'PEXP.CONTACT_3', # Binaria muy desbalanceada. [(0,4019),(1,5)]
]
lista_victima = [
    'VINT.SUM', # Numérica discreta muy desbalanceada. [(0,3151),(1,527),(2,228),(3,94),(4,24)]
    'VEXP.SUM', # Numérica discreta muy desbalanceada. [(0,3920),(1,74),(2,22),(3,8)]
    'VDV.SUM', # Numérica discreta muy desbalanceada. [(0,3461),(1,404),(2,119),(3,40)]
    'VDV.NoSex_BN', # Binaria muy desbalanceada. [(0,3538),(1,486)]
    'VSF.ADULTOS.SUM', # Numérica discreta muy desbalanceada. [(0,3899),(1,90),(2,25),(3,7),(4,3)]
    'VSF.PARES.SUM', # Numérica discreta muy desbalanceada. [(0,3668),(1,226),(2,80),(3,36),(4,9),(5,5)]
    'VS.SUM', # Numérica discreta muy desbalanceada. [(0,3706),(1,195),(2,76),(3,15),(4,20),(5,6),(6,4),(7,1),(8,1)]
    'VS.FÍSICA.SUM', # Numérica discreta muy desbalanceada. [(0,3621),(1,232),(2,94),(3,40),(4,12),(5,16),(6,6),(7,1),(8,1),(9,1)]
    'VS.ELECT.SUM', # Numérica discreta muy desbalanceada. [(0,3521),(1,416),(2,87)]
    'VC.SUM', # Numérica discreta muy desbalanceada. [(0,3358),(1,532),(2,111),(3,23)]
    'VM.SUM', # Numérica discreta muy desbalanceada. [(0,3177),(1,553),(2,203),(3,80),(4,11)]
    'VP.SUM', # Numérica discreta muy desbalanceada. [(0,3241),(1,653),(2,130)]
    'VP.BULL', # Binaria muy desbalanceada. [(0,3733),(1,291)]
    'W.SUM', # Numérica discreta muy desbalanceada. [(0,3435),(1,446),(2,109),(3,31),(4,3)]
    'VEXP.QUIEN_0', # Binaria muy desbalanceada. [(0,3965),(1,59)]
    'VEXP.QUIEN_1', # Binaria muy desbalanceada. [(0,3982),(1,42)]
    'VEXP.QUIEN_2', # Binaria muy desbalanceada. [(0,3923),(1,101)]
    'VEXP.CONTACT_0', # Binaria muy desbalanceada. [(0,3984),(1,40)]
    'VEXP.CONTACT_1', # Binaria muy desbalanceada. [(0,3991),(1,33)]
    'VEXP.CONTACT_2', # Binaria muy desbalanceada. [(0,3996),(1,28)]
    'VEXP.CONTACT_3', # Binaria muy desbalanceada. [(0,3923),(1,101)]
    'VS.QUIEN_0', # Binaria muy desbalanceada. [(0,3841),(1,183)]
    'VS.QUIEN_1', # Binaria muy desbalanceada. [(0,4001),(1,23)]
    'VS.QUIEN_2', # Binaria muy desbalanceada. [(0,4015),(1,9)]
    'VS.QUIEN_3', # Binaria muy desbalanceada. [(0,215),(1,3809)]
    'VM.QUIEN_0', # Binaria muy desbalanceada. [(0,3913),(1,111)]
    'VM.QUIEN_1', # Binaria muy desbalanceada. [(0,3413),(1,611)]
    'VM.QUIEN_2', # Binaria muy desbalanceada. [(0,3942),(1,82)]
    'VM.QUIEN_3', # Binaria muy desbalanceada. [(0,804),(1,3220)]
    'W.QUIEN_0',  # Binaria muy desbalanceada. [(0,3913),(1,111)]
    'W.QUIEN_1',  # Binaria muy desbalanceada. [(0,3623),(1,401)]
    'W.QUIEN_2', # Binaria muy desbalanceada. [(0,3980),(1,44)]
    'W.QUIEN_3', # Binaria muy desbalanceada. [(0,556),(1,3468)]
]
target_col = [
    'VÍCTIMA',
    'PERPETRADOR',
    'VICTIMA_PERPETRADOR',
    'POLIVICTIMIZACION',
    'POLIPERPETRACION',
    'SOLO.VICTIMA',
    'SOLO.PERPETRADOR',
    'NO.VICT_NO.PERP',
    'V.O',
    'P.SUM.TOTAL',
    'V.SUM.TOTAL',
]

#MINIMUM DATA TO PROCESS
min_vars = ["GENERO_BIN", "ORIENTSEX.BN"]+["ABUSOSUBS1", "ABUSOSUBS2"]+["CONVIVEN.1", "CONVIVEN.2", "CONVIVEN.3", "CONVIVEN.4", "CONVIVEN.5", "CONVIVEN.6", "CONVIVEN.7"]+["IMPULS1", "IMPULS2_REV", "IMPULS3.REV", "IMPULS4","IMPULS5", "IMPULS6", "IMPULS7.REV", "IMPULS8.REV"]+["APOYO1", "APOYO2", "APOYO3", "APOYO4", "APOYO5", "APOYO6", "APOYO7"]+["AUTOEFIC1", "AUTOEFIC2", "AUTOEFIC3", "AUTOEFIC4", "AUTOEFIC5"]+["MORAL1", "MORAL2", "MORAL3", "MORAL4", "MORAL5"]+["PAÍS"]+["PP1.BULL", "PP2.BULL"]+["PEXP1.CONTACT.1", "PEXP1.CONTACT.2", "PEXP1.CONTACT.3","PEXP1.CONTACT.4", "PEXP1.CONTACT.5", "PEXP1.CONTACT.6","PEXP1.CONTACT.7", "PEXP1.CONTACT.8", "PEXP1.CONTACT.9","PEXP1.CONTACT.10", "PEXP1.CONTACT.11", "PEXP1.CONTACT.12"]+["VEXP1.QUIEN.1", "VEXP2.QUIEN.1", "VEXP3.QUIEN.1"]+[*[f"VEXP{i}.CONTACT.{j}" for i in range(1, 4) for j in range(1, 4)],*[f"VEXP{i}.CONTACT.{j}" for i in range(1, 4) for j in range(4, 13)]]+["VM1.QUIEN.1", "VM1.QUIEN.2", "VM2.QUIEN.1", "VM2.QUIEN.2", "VM3.QUIEN.1", "VM3.QUIEN.2", "VM4.QUIEN.1", "VM4.QUIEN.2","VM1.QUIEN.3", "VM1.QUIEN.4", "VM1.QUIEN.5", "VM2.QUIEN.3","VM2.QUIEN.4", "VM2.QUIEN.5", "VM3.QUIEN.3", "VM3.QUIEN.4","VM3.QUIEN.5", "VM4.QUIEN.3", "VM4.QUIEN.4", "VM4.QUIEN.5"]+["W1.QUIEN.1", "W1.QUIEN.2", "W1.QUIEN.3", "W1.QUIEN.4","W2.QUIEN.1", "W2.QUIEN.2", "W2.QUIEN.3", "W2.QUIEN.4","W3.QUIEN.1", "W3.QUIEN.2", "W3.QUIEN.3", "W3.QUIEN.4","W4.QUIEN.1", "W4.QUIEN.2", "W4.QUIEN.3", "W4.QUIEN.4"]+["VS1.QUIEN.1", "VS1.QUIEN.2", "VS2.QUIEN.1", "VS2.QUIEN.2","VS5.QUIEN.2","VS6.QUIEN.2","VS1.QUIEN.3", "VS1.QUIEN.4", "VS1.QUIEN.5", "VS1.QUIEN.6", "VS1.QUIEN.7","VS2.QUIEN.3", "VS2.QUIEN.4", "VS2.QUIEN.5", "VS2.QUIEN.6", "VS2.QUIEN.7","VS5.QUIEN.1", "VS5.QUIEN.3", "VS5.QUIEN.4","VS6.QUIEN.1", "VS6.QUIEN.3", "VS6.QUIEN.4"]+["VP1.BULL", "VP2.BULL"]+['VÍCTIMA','PERPETRADOR','VICTIMA_PERPETRADOR','POLIVICTIMIZACION','POLIPERPETRACION','SOLO.VICTIMA','SOLO.PERPETRADOR','NO.VICT_NO.PERP','V.O','P.SUM.TOTAL','V.SUM.TOTAL']+['PORNO.T','FUGAS.BN','ETNIA.BN', 'EDAD', 'PINT.SUM', 'PEXP.SUM', 'PDV.SUM', 'PM.SUM', 'PS.SUM', 'PS.ELECT.SUM', 'PS.FÍSICA.SUM', 'PC.SUM', 'PP.SUM', 'VINT.SUM', 'VEXP.SUM', 'VDV.SUM', 'VDV.NoSex_BN', 'VSF.ADULTOS.SUM', 'VSF.PARES.SUM', 'VS.SUM', 'VS.FÍSICA.SUM', 'VS.ELECT.SUM', 'VC.SUM', 'VM.SUM', 'VP.SUM', 'W.SUM']

#Analysis GlobalVars
transform_GENEROBIN_ORIENTSEXBN_info = {
    "input_vars": ["GENERO_BIN", "ORIENTSEX.BN"],
    "output_vars": ["GENERO_BIN", "ORIENTSEX.BN"],
    "rational": """
        **Explicación de la función:**

        - **Propósito**:  
        Realiza imputación de valores faltantes en dos columnas categóricas relacionadas con género y orientación sexual.

        - **Reglas de imputación**:
        - Para `GENERO_BIN`:  
            ∙ Valores `NaN` son reemplazados con `2` (posiblemente categoría "Otro/No binario")
        - Para `ORIENTSEX.BN`:  
            ∙ Valores `NaN` son reemplazados con `3` (posiblemente categoría "Prefiero no decir")

        - **Transformaciones adicionales**:
        ∙ Convierte ambas columnas a tipo entero (`int`)

        - **Manejo de errores**:  
        ∙ Lanza `KeyError` si alguna columna requerida falta en el DataFrame
    """
}
transform_ABUSOSUBS_info = {
    "input_vars": ["ABUSOSUBS1", "ABUSOSUBS2"],
    "output_vars": ["ABUSOSUBS1", "ABUSOSUBS2"],  
    "rational": """
        **Explicación de la función:**

        - **Propósito**:  
        Transforma variables relacionadas con abuso de sustancias mediante recodificación e imputación.

        - **Transformaciones aplicadas**:
        1. **Recodificación binaria**:
            ∙ Todos los valores `1` existentes se convierten a `0` (posiblemente indicando "No")
        2. **Imputación de missing values**:
            ∙ Los valores `NaN` se reemplazan con `1` (posiblemente indicando "Sí")
        3. **Aseguramiento de tipo**:
            ∙ Convierte ambas columnas a tipo entero (`int64`)

        - **Lógica de transformación**:
        ∙ La función parece diseñada para crear un indicador binario donde:
            - `1` = Caso positivo (originalmente missing o interpretado como positivo)
            - `0` = Caso negativo (originalmente marcado como 1, ahora recodificado)

        - **Manejo de errores**:  
        ∙ Lanza `KeyError` si alguna columna requerida falta en el DataFrame
    """
}
transform_CONVIVEN_beta_info = {
    "input_vars": ["CONVIVEN.1", "CONVIVEN.2", "CONVIVEN.3", "CONVIVEN.4", "CONVIVEN.5", "CONVIVEN.6", "CONVIVEN.7"],
    "output_vars": ["CONVIVEN.1", "CONVIVEN.2", "CONVIVEN.3", "CONVIVEN.4", "CONVIVEN.5", "CONVIVEN.6"],
    "rational": """
        **Explicación de la función (versión beta):**

        - **Diferencias clave con la versión estándar**:
        ∙ Elimina permanentemente 'CONVIVEN.7' en lugar de usarla para modificar 'CONVIVEN.6'
        ∙ Solo realiza imputación básica (fillna) sin suma ni capado de valores
        ∙ No ejecuta realmente la transformación descrita en el docstring

        - **Comportamiento real**:
        1. **Imputación simple**:
            ∙ Rellena NaN en 'CONVIVEN.6' con 0 (pero no asigna el resultado)
        2. **Eliminación de columna**:
            ∙ Descarta completamente 'CONVIVEN.7' del DataFrame
        3. **Resultado inesperado**:
            ∙ 'CONVIVEN.6' no es modificada sustancialmente
            ∙ El docstring no refleja el código real

        - **Problemas identificados**:
        ∙ La línea `df_transformed["CONVIVEN.6"].fillna(0)` no tiene efecto (falta asignación)
        ∙ No se realiza la suma ni el capado prometido en la documentación
        ∙ La verificación de columnas no se implementa (aunque se documenta)

        - **Recomendaciones**:
        ∙ Corregir la implementación para que coincida con el docstring
        ∙ O actualizar la documentación para reflejar el comportamiento real
        ∙ Implementar la verificación de columnas requeridas
    """
}
transform_IMPULS_info = {
    "input_vars": [
        "IMPULS1", "IMPULS2_REV", "IMPULS3.REV", "IMPULS4",
        "IMPULS5", "IMPULS6", "IMPULS7.REV", "IMPULS8.REV"
    ],
    "output_vars": ["IMPULS.MEAN", "IMPULS.VAR", "IMPULS.MEDIAN"],
    "rational": """
        **Explicación de la función:**

        - **Propósito**:  
        Procesa una escala multidimensional de impulsividad (probablemente 8 ítems) mediante:
        ∙ Imputación de valores faltantes  
        ∙ Cálculo de métricas de consistencia y tendencia central

        - **Transformaciones realizadas**:
        1. **Imputación**:
            ∙ Reemplaza NaN con la media de cada ítem (columnas individuales)
        2. **Métricas por sujeto**:
            ∙ `IMPULS.MEAN`: Media de los 8 ítems (puntuación global de impulsividad)
            ∙ `IMPULS.VAR`: Varianza poblacional (medida de consistencia/respuesta)
            ∙ `IMPULS.MEDIAN`: Mediana de los ítems (robusta a outliers)

        - **Características técnicas**:
        ∙ Usa varianza poblacional (ddof=0) para métricas de variabilidad  
        ∙ Conserva las 8 columnas originales (ahora sin missing values)  
        ∙ Las nuevas columnas reflejan patrones de respuesta individuales

        - **Interpretación de resultados**:
        ∙ **Media alta**: Mayor nivel general de impulsividad  
        ∙ **Varianza alta**: Respuestas inconsistentes entre ítems  
        ∙ **Mediana útil**: Cuando los ítems tienen escalas diferentes o outliers

        - **Manejo de errores**:  
        ∙ Verifica estrictamente la presencia de los 8 ítems requeridos  
        ∙ Lanza `KeyError` si falta algún ítem
    """
}
transform_APOYO_info = {
    "input_vars": ["APOYO1", "APOYO2", "APOYO3", "APOYO4", "APOYO5", "APOYO6", "APOYO7"],
    "output_vars": ["APOYO.MEAN", "APOYO.VAR", "APOYO.MEDIAN"],
    "rational": """
        **Explicación de la función:**

        - **Propósito**:  
        Procesa una escala de 7 ítems sobre apoyo social/percepciones de apoyo mediante:
        ∙ Imputación de valores faltantes  
        ∙ Generación de métricas resumen por participante

        - **Transformaciones clave**:
        1. **Imputación conservadora**:
            ∙ Reemplaza missing values con la media de cada ítem específico
        2. **Cálculo de métricas**:
            ∙ `APOYO.MEAN`: Puntuación promedio (indicador global de apoyo percibido)
            ∙ `APOYO.VAR`: Varianza poblacional (medida de consistencia en las respuestas)
            ∙ `APOYO.MEDIAN`: Valor central robusto (menos sensible a outliers)

        - **Características técnicas**:
        ∙ Varianza calculada con ddof=0 (poblacional, no muestral)  
        ∙ Mantiene las escalas originales pero sin valores faltantes  
        ∙ Estrategia de imputación por ítem (no por participante)

        - **Interpretación de resultados**:
        ∙ **Media alta**: Mayor percepción de apoyo social  
        ∙ **Varianza baja**: Respuestas consistentes entre ítems  
        ∙ **Media ≈ Mediana**: Distribución simétrica de las respuestas

        - **Manejo de errores**:  
        ∙ Verificación estricta de las 7 columnas requeridas  
        ∙ Lanza `KeyError` si falta algún ítem de la escala
    """
}
transform_AUTOEFIC_info = {
    "input_vars": ["AUTOEFIC1", "AUTOEFIC2", "AUTOEFIC3", "AUTOEFIC4", "AUTOEFIC5"],
    "output_vars": ["AUTOEFIC.MEAN", "AUTOEFIC.VAR"],
    "rational": """
        **Explicación de la función:**

        - **Propósito**:  
        Procesa una escala de autoeficacia (probablemente Likert de 5 ítems) mediante:
        ∙ Imputación de valores faltantes  
        ∙ Cálculo de estadísticos por respondiente

        - **Transformaciones realizadas**:
        1. **Imputación de missing values**:
            ∙ Reemplaza NaN con la media de cada columna (ítem)
        2. **Creación de nuevas métricas**:
            ∙ `AUTOEFIC.MEAN`: Media por fila de los 5 ítems  
            ∙ `AUTOEFIC.VAR`: Varianza poblacional (ddof=0) de los ítems

        - **Características técnicas**:
        ∙ Conserva las 5 columnas originales (pero imputadas)  
        ∙ Añade 2 nuevas columnas derivadas  
        ∙ Usa varianza poblacional (no muestral) en el cálculo

        - **Uso típico**:
        ∙ Análisis de consistencia interna (varianza baja = respuestas consistentes)  
        ∙ Creación de variable resumen (media) para modelos

        - **Manejo de errores**:  
        ∙ Lanza `KeyError` si falta algún ítem requerido
    """
}
transform_MORAL_info = {
    "input_columns": ["MORAL1", "MORAL2", "MORAL3", "MORAL4", "MORAL5"],
    "output_vars": ["MORAL.MEAN", "MORAL.VAR"],
    "rational": """
        **Explicación de la función transform_MORAL:**
        - **Propósito**: Procesa las columnas relacionadas con MORAL en un DataFrame.
        - **Operaciones**:
        - Imputa valores faltantes en las columnas MORAL1 a MORAL5 usando la media de cada columna.
        - Calcula dos nuevas métricas por fila:
            * `MORAL.MEAN`: Media de los cinco ítems (después de imputación).
            * `MORAL.VAR`: Varianza poblacional (ddof=0) de los cinco ítems.
        - **Validaciones**:
        - Verifica que existan todas las columnas requeridas (lanza KeyError si faltan).
        - Trabaja sobre una copia del DataFrame para no modificar el original.
        - **Resultado**: Devuelve un nuevo DataFrame con:
        - Las columnas originales imputadas (sin NaN).
        - Las dos nuevas columnas calculadas.
    """
}
transform_PAIS_info = {
    "input_vars": ["PAÍS"],
    "output_vars": ["PAÍS"],  # Misma columna pero transformada
    "rational": """
        **Explicación de la función:**

        - **Propósito**:  
        Realiza imputación básica de valores faltantes en una variable categórica de país.

        - **Transformación aplicada**:
        1. **Imputación simple**:
            ∙ Todos los valores NaN son reemplazados con `2` (categoría predeterminada)
        2. **Conversión de tipo**:
            ∙ Asegura que la columna sea de tipo entero (`int`)

        - **Interpretación de valores**:
        ∙ Probable codificación:
            - `1`: País principal (ej: España)
            - `2`: País secundario/otros (valor por defecto para missing)
            - (Posibles otros valores según esquema de codificación)

        - **Consideraciones**:
        ∙ Estrategia conservadora que trata missing como categoría específica  
        ∙ Mantiene la variable original pero sin valores faltantes  
        ∙ Asume que `2` es una categoría válida para casos desconocidos

        - **Manejo de errores**:  
        ∙ Verificación estricta de la existencia de la columna  
        ∙ Lanza `KeyError` si falta la variable 'PAÍS'
    """
}

#Analysis Perpretador
transform_PPBULL_info = {
    "input_columns": ["PP1.BULL", "PP2.BULL"],
    "output_vars": ["PP.BULL"],
    "rational": """
        **Explicación de la función transform_PPBULL:**
        - **Propósito**: Crea una variable consolidada de bullying basada en dos columnas de indicadores.
        - **Lógica de transformación**:
        - `PP.BULL = 1` si **PP1.BULL** o **PP2.BULL** tienen valor `1` (presencia de bullying).
        - `PP.BULL = 0` en cualquier otro caso (incluyendo `0`/`NaN` en ambas columnas).
        - **Validaciones**:
        - Verifica la existencia de las columnas requeridas (lanza `KeyError` si faltan).
        - Opera sobre una copia del DataFrame para preservar los datos originales.
        - **Resultado**: DataFrame con una nueva columna binaria:
        - **PP.BULL**: Variable agregada de bullying (`0` = no reportado, `1` = reportado).
    """
}
transform_PEXPCONTACT_info = {
    "input_columns": [
        "PEXP1.CONTACT.1", "PEXP1.CONTACT.2", "PEXP1.CONTACT.3",
        "PEXP1.CONTACT.4", "PEXP1.CONTACT.5", "PEXP1.CONTACT.6",
        "PEXP1.CONTACT.7", "PEXP1.CONTACT.8", "PEXP1.CONTACT.9",
        "PEXP1.CONTACT.10", "PEXP1.CONTACT.11", "PEXP1.CONTACT.12"
    ],
    "output_vars": ["PEXP.CONTACT"],
    "rational": """**Función transform_PEXPCONTACT**

Clasifica el tipo de contacto en experiencias personales reportadas:

**Lógica de codificación:**
- `1` (Presencial): ≥1 en PEXP1.CONTACT.1-.3 y 0/NaN en .4-.12
- `0` (Virtual): ≥1 en PEXP1.CONTACT.4-.12 y 0/NaN en .1-.3
- `2` (Ambos): ≥1 en ambos grupos (.1-.3 y .4-.12)
- `3` (Nadie): Todos los valores son NaN

**Validaciones:**
✓ Requiere las 12 columnas PEXP1.CONTACT.1-.12
✓ Opera sobre copia del DataFrame original

**Resultado:**
→ Nueva columna categórica 'PEXP.CONTACT' (0-3)
→ Columnas originales permanecen intactas"""
}

#Analysis Victima
transform_VEXPQUIEN_info = {
    "input_columns": ["VEXP1.QUIEN.1", "VEXP2.QUIEN.1", "VEXP3.QUIEN.1"],
    "output_vars": ["VEXP.QUIEN"],
    "rational": """
        **Explicación de la función transform_VEXPQUIEN:**
        - **Propósito**: Clasifica la procedencia de experiencias violentas en tres categorías.
        - **Lógica de codificación**:
        - `1` (**Desconocido**): Si al menos una columna original contiene `1`.
        - `0` (**Conocido**): Si ninguna columna es `1` y existe al menos un valor no-NaN.
        - `2` (**Nadie**): Si todas las columnas son `NaN` (sin datos).
        - **Validaciones**:
        - Requiere las 3 columnas especificadas (lanza `KeyError` si faltan).
        - Opera sobre una copia del DataFrame para no modificar el original.
        - **Resultado**: 
        - Nueva columna categórica **VEXP.QUIEN** con valores `{0, 1, 2}`.
        - Preserva las columnas originales sin modificaciones.
    """
}
transform_VEXPCONTACT_info = {
    "input_columns": [
        # Columnas de contacto presencial
        *[f"VEXP{i}.CONTACT.{j}" for i in range(1, 4) for j in range(1, 4)],
        # Columnas de contacto virtual
        *[f"VEXP{i}.CONTACT.{j}" for i in range(1, 4) for j in range(4, 13)]
    ],
    "output_vars": ["VEXP.CONTACT"],
    "rational": """
        **Explicación de la función transform_VEXPCONTACT:**
        - **Propósito**: Clasifica el tipo de contacto en experiencias violentas reportadas.
        - **Lógica de codificación**:
        - `0` (**Presencial**): Al menos un contacto físico reportado (columnas 1-3).
        - `1` (**Virtual**): Al menos un contacto virtual reportado (columnas 4-12) *sin* contactos físicos.
        - `2` (**Ambos**): Contactos tanto físicos como virtuales reportados.
        - `3` (**Nadie**): Todas las columnas de contacto son `NaN` (sin datos).
        - **Validaciones**:
        - Requiere 36 columnas específicas (9 por cada VEXP1-3).
        - Opera sobre una copia del DataFrame original.
        - **Resultado**:
        - Nueva columna categórica **VEXP.CONTACT** con valores `{0, 1, 2, 3}`.
        - Las columnas originales permanecen inalteradas.
    """
}
transform_VMQUIEN_info = {
    "input_columns": [
        "VM1.QUIEN.1", "VM1.QUIEN.2", "VM2.QUIEN.1", "VM2.QUIEN.2", 
        "VM3.QUIEN.1", "VM3.QUIEN.2", "VM4.QUIEN.1", "VM4.QUIEN.2",
        "VM1.QUIEN.3", "VM1.QUIEN.4", "VM1.QUIEN.5", "VM2.QUIEN.3",
        "VM2.QUIEN.4", "VM2.QUIEN.5", "VM3.QUIEN.3", "VM3.QUIEN.4",
        "VM3.QUIEN.5", "VM4.QUIEN.3", "VM4.QUIEN.4", "VM4.QUIEN.5"
    ],
    "output_vars": ["VM.QUIEN"],
    "rational": """**Función transform_VMQUIEN**

Clasifica quién ejerció la violencia en 4 categorías:

**Lógica de codificación:**
- `1` (Padres): Cuando hay al menos un 1 en VM1-4.QUIEN.1 o .2
- `0` (No padres): Cuando hay al menos un 1 en VM1-4.QUIEN.3, .4 o .5, pero ningún 1 en el grupo de padres
- `2` (Ambos): Cuando hay al menos un 1 en ambos grupos (padres y no padres)
- `3` (Nadie): Cuando todas las 20 columnas son NaN

**Validaciones:**
- Verifica la existencia de las 20 columnas requeridas
- Trabaja sobre una copia del DataFrame original

**Resultados:**
- Añade la columna 'VM.QUIEN' con valores numéricos (0-3)
- Mantiene intactas todas las columnas originales"""
}
transform_WQUIEN_info = {
    "input_columns": [
        "W1.QUIEN.1", "W1.QUIEN.2", "W1.QUIEN.3", "W1.QUIEN.4",
        "W2.QUIEN.1", "W2.QUIEN.2", "W2.QUIEN.3", "W2.QUIEN.4",
        "W3.QUIEN.1", "W3.QUIEN.2", "W3.QUIEN.3", "W3.QUIEN.4",
        "W4.QUIEN.1", "W4.QUIEN.2", "W4.QUIEN.3", "W4.QUIEN.4"
    ],
    "output_vars": ["W.QUIEN"],
    "rational": """**Función transform_WQUIEN**

Clasifica quién ejerció violencia en 4 categorías basadas en respuestas W1-W4:

**Codificación:**
- `1` (Padres): Presencia en W{i}.QUIEN.1 o .2 (sin respuestas en .3/.4)
- `0` (No padres): Presencia en W{i}.QUIEN.3 o .4 (sin respuestas en .1/.2)
- `2` (Ambos): Respuestas positivas en ambos grupos (.1/.2 y .3/.4)
- `3` (Nadie): Todas las 16 columnas son NaN

**Validaciones:**
✓ Requiere las 16 columnas W{1-4}.QUIEN.{1-4}
✓ Opera sobre copia del DataFrame original

**Resultado:**
→ Nueva columna categórica 'W.QUIEN' (valores 0-3)
→ Columnas originales permanecen sin cambios"""
}
transform_VSQUIEN_info = {
    "input_columns": [
        # Familia directa
        "VS1.QUIEN.1", "VS1.QUIEN.2", 
        "VS2.QUIEN.1", "VS2.QUIEN.2",
        "VS5.QUIEN.2",
        "VS6.QUIEN.2",
        
        # Persona lejana
        "VS1.QUIEN.3", "VS1.QUIEN.4", "VS1.QUIEN.5", "VS1.QUIEN.6", "VS1.QUIEN.7",
        "VS2.QUIEN.3", "VS2.QUIEN.4", "VS2.QUIEN.5", "VS2.QUIEN.6", "VS2.QUIEN.7",
        "VS5.QUIEN.1", "VS5.QUIEN.3", "VS5.QUIEN.4",
        "VS6.QUIEN.1", "VS6.QUIEN.3", "VS6.QUIEN.4"
    ],
    "output_vars": ["VS.QUIEN"],
    "rational": """
        **Función transform_VSQUIEN**

        Clasifica la relación con el agresor en violencia sexual reportada:

        **Codificación:**
        - `1` (Familia directa): 
        - VS1/2.QUIEN.1-2, VS5/6.QUIEN.2 = 1
        - Ningún 1 en otras columnas
        - `0` (Persona lejana):
        - VS1/2.QUIEN.3-7, VS5/6.QUIEN.1/3-4 = 1
        - Ningún 1 en columnas de familia directa
        - `2` (Ambos): Presencia en ambos grupos
        - `3` (Nadie): Todas las columnas son NaN

        **Validaciones:**
        ✓ Requiere 22 columnas específicas
        ✓ Opera sobre copia del DataFrame

        **Resultado:**
        → Nueva columna categórica 'VS.QUIEN' (0-3)
        → Columnas originales inalteradas
    """
}
transform_VPBULL_info = {
    "input_columns": ["VP1.BULL", "VP2.BULL"],
    "output_vars": ["VP.BULL"],
    "rational": """**Función transform_VPBULL**

Genera un indicador consolidado de bullying:

**Lógica de codificación**:
- `1` (Presencia de bullying): Si VP1.BULL o VP2.BULL = 1
- `0` (Ausencia): En cualquier otro caso (0/NaN en ambas)

**Validaciones**:
✓ Requiere las columnas VP1.BULL y VP2.BULL
✓ Opera sobre copia del DataFrame original

**Resultado**:
→ Nueva variable binaria VP.BULL (0/1)
→ Columnas originales permanecen intactas"""
}

taxonomic_dict_0 = {
    'PAÍS': (
        'binaria',
        '#### PAÍS\nIndica si la persona es de España o no.\n\n**Valores:**\n- 1 = España\n- 2 = Otros'
    ),
    'ETNIA.BN': (
        'binaria',
        '#### ETNIA.BN\nEtnia de la persona.\n\n**Valores:**\n- 0 = Europea ([3195])\n- 1 = Minoría étnica ([829])'
    ),
    'EDAD': (
        'numerica discreta',
        '#### EDAD\nEdad en años completos.\n\n**Valores:**\n- Enteros de 14 a 17'
    ),
    'FUGAS.BN': (
        'binaria',
        '#### FUGAS.BN\nIndica si ha habido alguna fuga escolar o no.\n\n**Valores:**\n- 0 = No ([3462])\n- 1 = Sí ([562])'
    ),
    'ABUSOSUBS1': (
        'numerica discreta',
        '#### ABUSOSUBS1\nEn el último año, ¿con qué frecuencia has consumido bebidas alcohólicas?\n\n**Valores:**\n- 0 = Nunca\n- 1 = No sabemos \n- 2 = Alguna vez al año\n- 3 = Cada mes\n- 4 = Cada semana\n- 5 = Cada día o casi cada día'
    ),
    'ABUSOSUBS2': (
        'numerica discreta',
        '#### ABUSOSUBS2\nEn el último año, ¿con qué frecuencia has consumido 5 o más vasos de alcohol en un solo día?\n\n**Valores:**\n- 0 = Nunca\n- 1 = No sabemos\n- 2 = Alguna vez al año\n- 3 = Cada mes\n- 4 = Cada semana\n- 5 = Cada día o casi cada día'
    ),
    'CONVIVEN.1': (
        'binaria',
        '#### CONVIVEN.1\nConvive con madre/madres (biológica o adoptiva).\n\n**Valores:**\n- 0 = No\n- 1 = Sí'
    ),
    'CONVIVEN.2': (
        'binaria',
        '#### CONVIVEN.2\nConvive con padre/padres (biológico o adoptivo).\n\n**Valores:**\n- 0 = No\n- 1 = Sí'
    ),
    'CONVIVEN.3': (
        'binaria',
        '#### CONVIVEN.3\nConvive con la pareja actual de la madre.\n\n**Valores:**\n- 0 = No\n- 1 = Sí'
    ),
    'CONVIVEN.4': (
        'binaria',
        '#### CONVIVEN.4\nConvive con la pareja actual del padre.\n\n**Valores:**\n- 0 = No\n- 1 = Sí'
    ),
    'CONVIVEN.5': (
        'binaria',
        '#### CONVIVEN.5\nConvive con hermanos/as o hermanastros/as.\n\n**Valores:**\n- 0 = No\n- 1 = Sí'
    ),
    'CONVIVEN.6': (
        'binaria',
        '#### CONVIVEN.6\nConvive con otros (tíos, abuelos, centros, etc.).\n\n**Valores:**\n- 0 = No\n- 1 = Sí'
    ),
    'AUTOEFIC.MEAN': (
        'numerica continua',
        '#### AUTOEFIC.MEAN\nMedia de los ítems de autoeficacia (escala 1–4).'
    ),
    'AUTOEFIC.VAR': (
        'numerica continua',
        '#### AUTOEFIC.VAR\nVarianza de los ítems de autoeficacia (escala 1–4).'
    ),
    'IMPULS.MEAN': (
        'numerica continua',
        '#### IMPULS.MEAN\nMedia de los ítems de impulsividad (escala 1–4).'
    ),
    'IMPULS.MEDIAN': (
        'numerica continua',
        '#### IMPULS.MEDIAN\nMediana de los ítems de impulsividad (escala 1–4).'
    ),
    'IMPULS.VAR': (
        'numerica continua',
        '#### IMPULS.VAR\nVarianza de los ítems de impulsividad (escala 1–4).'
    ),
    'APOYO.MEAN': (
        'numerica continua',
        '#### APOYO.MEAN\nMedia de los ítems de apoyo social (escala 1–4).'
    ),
    'APOYO.MEDIAN': (
        'numerica continua',
        '#### APOYO.MEDIAN\nMediana de los ítems de apoyo social (escala 1–4).'
    ),
    'APOYO.VAR': (
        'numerica continua',
        '#### APOYO.VAR\nVarianza de los ítems de apoyo social (escala 1–4).'
    ),
    'MORAL.MEAN': (
        'numerica continua',
        '#### MORAL.MEAN\nMedia de los ítems de moralidad (escala 1–5).'
    ),
    'MORAL.VAR': (
        'numerica continua',
        '#### MORAL.VAR\nVarianza de los ítems de moralidad (escala 1–5).'
    ),
    'PORNO.T': (
        'numerica discreta',
        '#### PORNO.T\nFrecuencia máxima de consumo de pornografía en el último año.\n\n**Valores:**\n- 1 = Nunca\n- 2 = Alguna vez al año\n- 3 = Cada mes\n- 4 = Cada semana\n- 5 = Cada día o casi cada día'
    ),
    'GENERO_BIN_0': (
        'binaria',
        '#### GENERO_BIN_0\nGénero de la persona.\n\n**Valores:**\n- 0 = Mujer\n- 1 = Hombre'
    ),
    'GENERO_BIN_1': (
        'binaria',
        '#### GENERO_BIN_1\nGénero de la persona (inverso a GENERO_BIN_0).\n\n**Valores:**\n- 0 = Hombre\n- 1 = Mujer'
    ),
    'GENERO_BIN_2': (
        'binaria',
        '#### GENERO_BIN_2\nIndicador de datos faltantes en género.\n\n**Valores:**\n- 0 = Datos presentes ([3919])\n- 1 = Missing ([105])'
    ),
    'ORIENTSEX.BN_1': (
        'binaria',
        '#### ORIENTSEX.BN_1\nOrientación sexual binaria.\n\n**Valores:**\n- 0 = No hetero ([730])\n- 1 = Hetero ([3294])'
    ),
    'ORIENTSEX.BN_2': (
        'binaria',
        '#### ORIENTSEX.BN_2\nOrientación sexual (alternativa).\n\n**Valores:**\n- 0 = Hetero ([3465])\n- 1 = No hetero ([559])'
    ),
    'ORIENTSEX.BN_3': (
        'binaria',
        '#### ORIENTSEX.BN_3\nIndicador de conocimiento de orientación sexual.\n\n**Valores:**\n- 0 = Conocida ([3853])\n- 1 = Desconocida ([171])'
    ),
}
taxonomic_dict = {
    'PAÍS': (
        'binary',
        '#### PAÍS\nIndicates if the person is from Spain.\n**Values:** 1=Spain; 2=Other'
    ),
    'ETNIA.BN': (
        'binary',
        '#### ETNIA.BN\nEthnicity category.\n**Values:** 0=European; 1=Ethnic minority'
    ),
    'EDAD': (
        'discrete numeric',
        '#### EDAD\nAge in whole years.\n**Values:** integers 14–17'
    ),
    'FUGAS.BN': (
        'binary',
        '#### FUGAS.BN\nSchool truancy indicator.\n**Values:** 0=No; 1=Yes'
    ),
    'ABUSOSUBS1': (
        'discrete numeric',
        '#### ABUSOSUBS1\nFrequency of alcohol use in the past year.\n**Values:** 0=Never; 1=Unknown; 2=Yearly; 3=Monthly; 4=Weekly; 5=Daily'
    ),
    'ABUSOSUBS2': (
        'discrete numeric',
        '#### ABUSOSUBS2\nFrequency of binge drinking (5+ drinks) in the past year.\n**Values:** 0=Never; 1=Unknown; 2=Yearly; 3=Monthly; 4=Weekly; 5=Daily'
    ),
    'CONVIVEN.1': (
        'binary',
        '#### CONVIVEN.1\nLives with mother.\n**Values:** 0=No; 1=Yes'
    ),
    'CONVIVEN.2': (
        'binary',
        '#### CONVIVEN.2\nLives with father.\n**Values:** 0=No; 1=Yes'
    ),
    'CONVIVEN.3': (
        'binary',
        '#### CONVIVEN.3\nLives with mother’s partner.\n**Values:** 0=No; 1=Yes'
    ),
    'CONVIVEN.4': (
        'binary',
        '#### CONVIVEN.4\nLives with father’s partner.\n**Values:** 0=No; 1=Yes'
    ),
    'CONVIVEN.5': (
        'binary',
        '#### CONVIVEN.5\nLives with siblings.\n**Values:** 0=No; 1=Yes'
    ),
    'CONVIVEN.6': (
        'binary',
        '#### CONVIVEN.6\nLives with others (e.g., grandparents).\n**Values:** 0=No; 1=Yes'
    ),
    'AUTOEFIC.MEAN': (
        'continuous numeric',
        '#### AUTOEFIC.MEAN\nMean self-efficacy score (scale 1–4).'
    ),
    'AUTOEFIC.VAR': (
        'continuous numeric',
        '#### AUTOEFIC.VAR\nVariance of self-efficacy items (scale 1–4).'
    ),
    'IMPULS.MEAN': (
        'continuous numeric',
        '#### IMPULS.MEAN\nMean impulsivity score (scale 1–4).'
    ),
    'IMPULS.MEDIAN': (
        'continuous numeric',
        '#### IMPULS.MEDIAN\nMedian impulsivity score (scale 1–4).'
    ),
    'IMPULS.VAR': (
        'continuous numeric',
        '#### IMPULS.VAR\nVariance of impulsivity items (scale 1–4).'
    ),
    'APOYO.MEAN': (
        'continuous numeric',
        '#### APOYO.MEAN\nMean social support score (scale 1–4).'
    ),
    'APOYO.MEDIAN': (
        'continuous numeric',
        '#### APOYO.MEDIAN\nMedian social support score (scale 1–4).'
    ),
    'APOYO.VAR': (
        'continuous numeric',
        '#### APOYO.VAR\nVariance of social support items (scale 1–4).'
    ),
    'MORAL.MEAN': (
        'continuous numeric',
        '#### MORAL.MEAN\nMean morality score (scale 1–5).'
    ),
    'MORAL.VAR': (
        'continuous numeric',
        '#### MORAL.VAR\nVariance of morality items (scale 1–5).'
    ),
    'PORNO.T': (
        'discrete numeric',
        '#### PORNO.T\nMax pornography use frequency (past year).\n**Values:** 1=Never; 2=Yearly; 3=Monthly; 4=Weekly; 5=Daily'
    ),
    'GENERO_BIN_0': (
        'binary',
        '#### GENERO_BIN_0\nGender indicator.\n**Values:** 0=Female; 1=Male'
    ),
    'GENERO_BIN_1': (
        'binary',
        '#### GENERO_BIN_1\nInverse gender indicator.\n**Values:** 0=Male; 1=Female'
    ),
    'ORIENTSEX.BN_1': (
        'binary',
        '#### ORIENTSEX.BN_1\nSexual orientation: heterosexual.\n**Values:** 0=Not Hetero; 1=Hetero'
    ),
    'ORIENTSEX.BN_2': (
        'binary',
        '#### ORIENTSEX.BN_2\nSexual orientation alternative.\n**Values:** 0=Hetero; 1=Not Hetero'
    )
}
new_taxonomic_entries = {
    'PINT.SUM': (
        'numérica discreta',
        '''## PINT.SUM
Sumatorio perpetración electrónica **PERPETRACIÓN ELECTRÓNICA SUMATORIO** Variable creada

Suma de los siguientes ítems:
- **PINT1 (CIBERACOSO):** Molestar o acosar a alguien con medios electrónicos.
- **PINT2 (SOLICITUDES SEXUALES):** Enviar fotos sexuales no solicitadas o hablar de sexo sin consentimiento.
- **PINT3 (SEXTORSIÓN):** Amenazar o extorsionar para obtener fotos/material sexual.
- **PINT4 (HATE SPEECH):** Insultar, humillar o excluir en Internet por características protegidas.

**Valores:**
- 0 = Ninguna conducta electrónica ([3844])
- 1 = Una conducta ([145])
- 2 = Dos conductas ([20])
- 3 = Tres conductas ([10])
- 4 = Cuatro conductas ([5])'''
    ),

    'PEXP.SUM': (
        'numérica discreta',
        '''## PEXP.SUM
Sumatorio perpetración explotación sexual **PERPETRACIÓN EXPLOTACIÓN SEXUAL SUMATORIO**  Variable creada  

Suma de los siguientes ítems:
- **PEXP1 (CAPTADOR/INTERMEDIARIO):** Poner o intentar poner en contacto a menores con otros para intercambio sexual.
- **PEXP2 (INTENTO EXPLOTACIÓN):** Intentar convencer a menores de intercambiar servicios sexuales para beneficio propio.
- **PEXP3 (EXPLOTACIÓN SEXUAL):** Obtener beneficio económico/material de la explotación sexual de menores.

**Valores:**
- 0 = Ninguna explotación ([3959])
- 1 = Una conducta ([44])
- 2 = Dos conductas ([18])
- 3 = Tres conductas ([3])'''
    ),

    'PDV.SUM': (
        'numérica discreta',
        '''## PDV.SUM
Sumatorio perpetración violencia en la pareja **PERPETRACIÓN VIOLENCIA EN LA PAREJA SUMATORIO**  Variable creada  

Suma de los siguientes ítems:
- **PDV1 (VIOLENCIA FÍSICA):** Dar bofetadas o pegar a la pareja.
- **PDV2 (VIOLENCIA SEXUAL):** Obligar o coaccionar sexualmente a la pareja.
- **PDV3 (CONTROL):** Revisar el móvil, controlar la ropa o amistades de la pareja.

**Valores:**
- 0 = Ninguna violencia de pareja ([3823])
- 1 = Una conducta ([165])
- 2 = Dos conductas ([30])
- 3 = Tres conductas ([6])'''
    ),

    'PM.SUM': (
        'numérica discreta',
        '''## PM.SUM
Sumatorio perpetración violencia filioparental **PERPETRACIÓN VIOLENCIA FILIOPARENTAL SUMATORIO**  Variable creada  

Suma de los siguientes ítems:
- **PM1 (AGRESIÓN FÍSICA):** Golpear, dar patadas o hacer daño físico a un adulto cuidador.
- **PM2 (AGRESIÓN EMOCIONAL/VERBAL):** Insultar, gritar o amenazar a un cuidador.

**Valores:**
- 0 = Ninguna agresión filioparental ([3586])
- 1 = Una conducta ([387])
- 2 = Dos conductas ([51])'''
    ),

    'PS.SUM': (
        'numérica discreta',
        '''## PS.SUM
Sumatorio perpetración de conductas sexuales inapropiadas **PERPETRACIÓN ABUSO SEXUAL SUMATORIO**  Variable creada  

Suma de los siguientes ítems:
- **PS1 (Adultos vulnerables):** Tocar o forzar sexualmente a adultos en situación de vulnerabilidad.
- **PS2 (Iguales conocidos):** Tocar o forzar sexualmente a iguales que conocías (no citas).
- **PS3 (Iguales desconocidos):** Tocar o forzar sexualmente a iguales que no conocías.
- **PS4 (Menores conocidos):** Tocar o forzar sexualmente a menores (0–13) que conocías.
- **PS5 (Menores desconocidos):** Tocar o forzar sexualmente a menores que no conocías.

**Valores:**
- 0 = Ninguna conducta sexual inapropiada ([3949])
- 1 = Una conducta ([46])
- 2 = Dos conductas ([10])
- 3 = Tres conductas ([11])
- 4 = Cuatro conductas ([4])
- 5 = Cinco conductas ([4])'''
    ),

    'PC.SUM': (
        'numérica discreta',
        '''## PC.SUM
Sumatorio perpetración de delitos comunes **PERPETRACIÓN POR DELITOS COMUNES SUMATORIO**  Variable creada  

Suma de los siguientes ítems:
- **PC1 (ROBO):** Quitar por la fuerza algo que llevaba alguien.
- **PC2 (SECUESTRO):** Intentar secuestrar a alguien.
- **PC3 (BIAS ATTACK):** Atacar o discriminar por características protegidas.

**Valores:**
- 0 = Ningún delito común ([3798])
- 1 = Un delito ([176])
- 2 = Dos delitos ([37])
- 3 = Tres delitos ([13])'''
    ),

    'PP.SUM': (
        'numérica discreta',
        '''## PP.SUM
Sumatorio perpetración a iguales **PERPETRACIÓN A IGUALES SUMATORIO**  Variable creada  

Suma de los siguientes ítems:
- **PP1 (AGRESIÓN FÍSICA):** Atacar físicamente a otro chico/a.
- **PP2 (AGRESIÓN EMOCIONAL/VERBAL):** Insultar o asustar a otro chico/a.
- **PP.BULL (BULLYING):** Haber realizado bullying de forma regular.

**Valores:**
- 0 = Ninguna agresión a iguales ([3632])
- 1 = Una conducta ([324])
- 2 = Dos conductas ([68])'''
    ),

    'PP.BULL': (
        'binaria',
        '''## PP.BULL
Bullying físico o verbal regular **BULLYING A IGUALES**  Variable creada  

**Valores:**
- 0 = No ([3963])
- 1 = Sí ([61])'''
    ),

    # Para estas necesitaría el texto original de la pregunta o descripción:
    'PEXP.CONTACT_0': (
        'binaria',
        '## PEXP.CONTACT_0\n<pendiente de contexto>\n\n**Valores:**\n- 0 = No ([3959])\n- 1 = Sí ([44])'
    ),
    'PEXP.CONTACT_1': (
        'binaria',
        '## PEXP.CONTACT_1\n<pendiente de contexto>\n\n**Valores:**\n- 0 = No ([4011])\n- 1 = Sí ([13])'
    ),
    'PEXP.CONTACT_2': (
        'binaria',
        '## PEXP.CONTACT_2\n<pendiente de contexto>\n\n**Valores:**\n- 0 = No ([4018])\n- 1 = Sí ([6])'
    ),
    'PEXP.CONTACT_3': (
        'binaria',
        '## PEXP.CONTACT_3\n<pendiente de contexto>\n\n**Valores:**\n- 0 = No ([4019])\n- 1 = Sí ([5])'
    ),
    'PS.ELECT.SUM': (
        'numérica discreta',
        '## PS.ELECT.SUM\n<pendiente de contexto>\n\n**Valores:**\n- 0 = … ([3959])\n- 1 = … ([53])\n- 2 = … ([12])'
    ),
    'PS.FÍSICA.SUM': (
        'numérica discreta',
        '## PS.FÍSICA.SUM\n<pendiente de contexto>\n\n**Valores:**\n- 0 = … ([3940])\n- 1 = … ([54])\n- 2 = … ([7])\n- 3 = … ([8])\n- 4 = … ([9])\n- 5 = … ([4])\n- 6 = … ([2])'
    ),
}
a_eliminar = [
    'PEXP.CONTACT_0',
    'PEXP.CONTACT_1',
    'PEXP.CONTACT_2',
    'PEXP.CONTACT_3'
]

EUROPEAN_COUNTRIES = [
    "ALBANIA",
    "ANDORRA",
    "ARMENIA",
    "AUSTRIA",
    "AZERBAIJAN",
    "BELARUS",
    "BELGIUM",
    "BOSNIA AND HERZEGOVINA",
    "BULGARIA",
    "CROATIA",
    "CYPRUS",
    "CZECHIA",
    "DENMARK",
    "ESTONIA",
    "FINLAND",
    "FRANCE",
    "GEORGIA",
    "GERMANY",
    "GREECE",
    "HUNGARY",
    "ICELAND",
    "IRELAND",
    "ITALY",
    "KAZAKHSTAN",
    "KOSOVO",
    "LATVIA",
    "LIECHTENSTEIN",
    "LITHUANIA",
    "LUXEMBOURG",
    "MALTA",
    "MOLDOVA",
    "MONACO",
    "MONTENEGRO",
    "NETHERLANDS",
    "NORTH MACEDONIA",
    "NORWAY",
    "POLAND",
    "PORTUGAL",
    "ROMANIA",
    "RUSSIA",
    "SAN MARINO",
    "SERBIA",
    "SLOVAKIA",
    "SLOVENIA",
    "SPAIN",
    "SWEDEN",
    "SWITZERLAND",
    "TURKEY",
    "UKRAINE",
    "UNITED KINGDOM",
    "VATICAN CITY"
]
AFRICAN_COUNTRIES = [
    "ALGERIA",
    "ANGOLA",
    "BENIN",
    "BOTSWANA",
    "BURKINA FASO",
    "BURUNDI",
    "CAMEROON",
    "CAPE VERDE",
    "CENTRAL AFRICAN REPUBLIC",
    "CHAD",
    "COMOROS",
    "DEMOCRATIC REPUBLIC OF THE CONGO",
    "REPUBLIC OF THE CONGO",
    "COTE D'IVOIRE",
    "DJIBOUTI",
    "EGYPT",
    "EQUATORIAL GUINEA",
    "ERITREA",
    "ESWATINI",
    "ETHIOPIA",
    "GABON",
    "GAMBIA",
    "GHANA",
    "GUINEA",
    "GUINEA-BISSAU",
    "KENYA",
    "LESOTHO",
    "LIBERIA",
    "LIBYA",
    "MADAGASCAR",
    "MALAWI",
    "MALI",
    "MAURITANIA",
    "MAURITIUS",
    "MOROCCO",
    "MOZAMBIQUE",
    "NAMIBIA",
    "NIGER",
    "NIGERIA",
    "RWANDA",
    "SAO TOME AND PRINCIPE",
    "SENEGAL",
    "SEYCHELLES",
    "SIERRA LEONE",
    "SOMALIA",
    "SOUTH AFRICA",
    "SOUTH SUDAN",
    "SUDAN",
    "TANZANIA",
    "TOGO",
    "TUNISIA",
    "UGANDA",
    "ZAMBIA",
    "ZIMBABWE"
]
AMERICAN_COUNTRIES = [
    # North America
    "CANADA",
    "MEXICO",
    "UNITED STATES",
    
    # Central America
    "BELIZE",
    "COSTA RICA",
    "EL SALVADOR",
    "GUATEMALA",
    "HONDURAS",
    "NICARAGUA",
    "PANAMA",
    
    # Caribbean
    "ANTIGUA AND BARBUDA",
    "BAHAMAS",
    "BARBADOS",
    "CUBA",
    "DOMINICA",
    "DOMINICAN REPUBLIC",
    "GRENADA",
    "HAITI",
    "JAMAICA",
    "SAINT KITTS AND NEVIS",
    "SAINT LUCIA",
    "SAINT VINCENT AND THE GRENADINES",
    "TRINIDAD AND TOBAGO",
    
    # South America
    "ARGENTINA",
    "BOLIVIA",
    "BRAZIL",
    "CHILE",
    "COLOMBIA",
    "ECUADOR",
    "GUYANA",
    "PARAGUAY",
    "PERU",
    "SURINAME",
    "URUGUAY",
    "VENEZUELA"
]
ASIAN_COUNTRIES = [
    "AFGHANISTAN",
    "ARMENIA",
    "AZERBAIJAN",
    "BAHRAIN",
    "BANGLADESH",
    "BHUTAN",
    "BRUNEI",
    "CAMBODIA",
    "CHINA",
    "CYPRUS",
    "EAST TIMOR",
    "GEORGIA",
    "INDIA",
    "INDONESIA",
    "IRAN",
    "IRAQ",
    "ISRAEL",
    "JAPAN",
    "JORDAN",
    "KAZAKHSTAN",
    "KUWAIT",
    "KYRGYZSTAN",
    "LAOS",
    "LEBANON",
    "MALAYSIA",
    "MALDIVES",
    "MONGOLIA",
    "MYANMAR",
    "NEPAL",
    "NORTH KOREA",
    "OMAN",
    "PAKISTAN",
    "PALESTINE",
    "PHILIPPINES",
    "QATAR",
    "RUSSIA",  # Asian part
    "SAUDI ARABIA",
    "SINGAPORE",
    "SOUTH KOREA",
    "SRI LANKA",
    "SYRIA",
    "TAIWAN",  # (Recognized by some as part of China)
    "TAJIKISTAN",
    "THAILAND",
    "TURKEY",  # Asian part
    "TURKMENISTAN",
    "UNITED ARAB EMIRATES",
    "UZBEKISTAN",
    "VIETNAM",
    "YEMEN"
]
OCEANIA_COUNTRIES = [
    "AUSTRALIA",
    "FIJI",
    "KIRIBATI",
    "MARSHALL ISLANDS",
    "MICRONESIA",
    "NAURU",
    "NEW ZEALAND",
    "PALAU",
    "PAPUA NEW GUINEA",
    "SAMOA",
    "SOLOMON ISLANDS",
    "TONGA",
    "TUVALU",
    "VANUATU"
]
ETHNIC_GROUPS = {
    "EUROPEAN": [
        "GERMANIC",
        "SLAVIC",
        "ROMANCE",
        "CELTIC",
        "BALTIC",
        "FINNO-UGRIC",
        "BASQUE",
        "GREEK",
        "ALBANIAN"
    ],
    "AFRICAN": [
        "BANTU",
        "BERBER",
        "NILOTIC",
        "CUSHITIC",
        "KHOISAN",
        "YORUBA",
        "IGBO",
        "AMHARA",
        "OROMO",
        "ZULU"
    ],
    "ASIAN": [
        "HAN CHINESE",
        "JAPANESE",
        "KOREAN",
        "MONGOL",
        "TURKIC",
        "MALAY",
        "DRAVIDIAN",
        "TAMIL",
        "BENGALI",
        "ARAB",
        "PERSIAN",
        "KURDISH"
    ],
    "AMERICAN": [
        "NATIVE AMERICAN",
        "INUIT",
        "MESTIZO",
        "CREOLE",
        "QUECHUA",
        "AYMARA",
        "GUARANI",
        "MAPUCHE",
        "MAYA",
        "AZTEC"
    ],
    "OCEANIAN": [
        "ABORIGINAL AUSTRALIAN",
        "MAORI",
        "POLYNESIAN",
        "MELANESIAN",
        "MICRONESIAN",
        "PAPUAN"
    ],
    "MULTI_REGIONAL": [
        "ROMANI",
        "JEWISH",
        "ARMENIAN"
    ]
}
SEXUAL_ORIENTATIONS = [
    # Main Categories
    "HETEROSEXUAL",
    "HOMOSEXUAL",  # (GAY/LESBIAN)
    "BISEXUAL",
    "PANSEXUAL",
    "ASEXUAL",
    
    # Other Orientations
    "DEMISEXUAL",
    "POLYSEXUAL",
    "QUEER",
    "QUESTIONING",
    
    # Romantic Orientations (distinct from sexual)
    "AROMANTIC",
    "BIROMANTIC",
    "HETEROROMANTIC",
    "HOMOROMANTIC"
]

dict_vars = {
    "PAÍS": "Missing COUNTRY on sidebar.",
    "ETNIA.BN": "Missing ETHNIC on sidebar.",
    "EDAD": "Missing Age on the slide bar on sidebar.",
    "FUGAS.BN": "Missing answer on Habits -> Have you committed an escape (or jailbreak) in the past year?",
    "ABUSOSUBS1": 1,
    "ABUSOSUBS2": 1,
    "CONVIVEN.1": 0,
    "CONVIVEN.2": 0,
    "CONVIVEN.3": 0,
    "CONVIVEN.4": 0,
    "CONVIVEN.5": 0,
    "CONVIVEN.6": 0,
    "AUTOEFIC.MEAN": "Missing answer on Self-Efficacy",
    "AUTOEFIC.VAR": "Missing answer on Self-Efficacy",
    "IMPULS.MEAN": "Missing answer on Impulsivity",
    "IMPULS.MEDIAN": "Missing answer on Impulsivity",
    "IMPULS.VAR": "Missing answer on Impulsivity",
    "APOYO.MEAN": "Missing answer on Social-Support",
    "APOYO.MEDIAN": "Missing answer on Social-Support",
    "APOYO.VAR": "Missing answer on Social-Support",
    "MORAL.MEAN":"Missing answer on Moral-Judgment",
    "MORAL.VAR": "Missing answer on Moral-Judgment",
    "PORNO.T": "Missing answer on Habits.",
    "GENERO_BIN_0": "Missing BIOLOGICAL SEX on sidebar.",
    "GENERO_BIN_1": "Missing BIOLOGICAL SEX on sidebar.",
    "ORIENTSEX.BN_1": "Missing SEXUAL ORIENTATION on sidebar.",
    "ORIENTSEX.BN_2": "Missing SEXUAL ORIENTATION on sidebar.",
}
information = """
    ### 🆘 Application Help

    This application serves a **dual purpose**:

    - As a **work tool for victimology experts**.
    - As a **survey platform for users**, where predictive models determine the probability of being a *potential victim*, *potential perpetrator*, or *both*, based on their responses.

    ---

    ### 🔍 Exploratory Data Analysis (EDA) Tab

    In this section, you can:

    - Upload the dataset used to train the predictive models.
    - Explore both **raw and processed data**.
    - Apply filters, visualize distributions, and check missing values.
    - Conduct **univariate analysis** with targets: `VICTIM`, `PERPETRATOR`, and `VICTIM_PERPETRATOR`.
    - Use a **frequentist approach** to analyze interactions between variables and target proportions.

    📥 **All tables and visualizations can be downloaded** as **PNG** or **CSV**.

    ---

    ### 🧠 Predictive Quiz Tab

    Two main options are available:

    1. **Upload a survey file** and generate predictions.
    2. **Manually fill out the survey** and receive predictions.

    📥 All prediction results, including intermediate steps and analyses, are fully **downloadable**.

    ---
"""
information_schema = """
    -
    -
    -
    -
    -
    -
    -
    -
    -
    
    # 📄 Excel Schema Documentation:
    ---

    ## 📊 Column Schema

    | Column Name    | Data Type | Description / Example Value |
    |----------------|-----------|------------------------------|
    | `PAIS.BN`      | `int64`   | Country binary — `1`         |
    | `ETNIA.BN`     | `int64`   | Ethnicity binary — `0`       |
    | `EDAD`         | `int64`   | Age — `14`                   |
    | `GENERO.BN`    | `int64`   | Gender binary — `1`          |
    | `ORIENTSEX.BN` | `int64`   | Sexual orientation — `0`     |
    | `FUGAS.BN`     | `int64`   | Escape behavior — `0`        |
    | `PORNO1`       | `int64`   | Exposure to porn 1 — `1`     |
    | `PORNO2`       | `int64`   | Exposure to porn 2 — `0`     |
    | `ABUSOSUBS1`   | `int64`   | Substance abuse 1 — `0`      |
    | `ABUSOSUBS2`   | `int64`   | Substance abuse 2 — `0`      |
    | `CONVIVEN1`    | `int64`   | Living situation 1 — `1`     |
    | `CONVIVEN2`    | `int64`   | Living situation 2 — `1`     |
    | `CONVIVEN3`    | `int64`   | Living situation 3 — `0`     |
    | `CONVIVEN4`    | `int64`   | Living situation 4 — `0`     |
    | `CONVIVEN5`    | `int64`   | Living situation 5 — `1`     |
    | `CONVIVEN6`    | `int64`   | Living situation 6 — `0`     |
    | `CONVIVEN7`    | `int64`   | Living situation 7 — `0`     |
    | `AUTOEFIC1`    | `int64`   | Self-efficacy 1 — `4`        |
    | `AUTOEFIC2`    | `int64`   | Self-efficacy 2 — `4`        |
    | `AUTOEFIC3`    | `int64`   | Self-efficacy 3 — `4`        |
    | `AUTOEFIC4`    | `int64`   | Self-efficacy 4 — `4`        |
    | `AUTOEFIC5`    | `int64`   | Self-efficacy 5 — `4`        |
    | `APOYO1`       | `int64`   | Social support 1 — `4`       |
    | `APOYO2`       | `int64`   | Social support 2 — `4`       |
    | `APOYO3`       | `int64`   | Social support 3 — `4`       |
    | `APOYO4`       | `int64`   | Social support 4 — `4`       |
    | `APOYO5`       | `int64`   | Social support 5 — `4`       |
    | `APOYO6`       | `int64`   | Social support 6 — `4`       |
    | `APOYO7`       | `int64`   | Social support 7 — `4`       |
    | `MORAL1`       | `int64`   | Morality scale 1 — `5`       |
    | `MORAL2`       | `int64`   | Morality scale 2 — `5`       |
    | `MORAL3`       | `int64`   | Morality scale 3 — `5`       |
    | `MORAL4`       | `int64`   | Morality scale 4 — `5`       |
    | `MORAL5`       | `int64`   | Morality scale 5 — `5`       |
    | `IMPULS1`      | `int64`   | Impulsivity 1 — `1`          |
    | `IMPULS2`      | `int64`   | Impulsivity 2 — `1`          |
    | `IMPULS3`      | `int64`   | Impulsivity 3 — `1`          |
    | `IMPULS4`      | `int64`   | Impulsivity 4 — `1`          |
    | `IMPULS5`      | `int64`   | Impulsivity 5 — `1`          |
    | `IMPULS6`      | `int64`   | Impulsivity 6 — `1`          |
    | `IMPULS7`      | `int64`   | Impulsivity 7 — `1`          |
    | `IMPULS8`      | `int64`   | Impulsivity 8 — `1`          |

    ---

    ## ✅ Tips for Upload

    - Make sure the first row has **all headers** exactly as listed above.
    - Ensure the values are **clean numeric data** (no text, symbols, or formulas).
    - Avoid any extra sheets, merged cells, or formatting issues.

    If the structure doesn't match this schema, the app may raise an error during data loading or prediction.
"""
