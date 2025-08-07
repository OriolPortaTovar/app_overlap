import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.io as pio

from typing import Dict, Tuple

def create_binary_plot_figure(df: pd.DataFrame, var: str) -> go.Figure:
    """
    Genera un objeto plotly.graph_objects.Figure de 1x3:
    1. Conteos de la variable binaria `var`, mostrando la proporción como texto.
    2. Conteos de las 4 combinaciones de `var` con 'VÍCTIMA', con proporción en hover.
    3. Conteos de las 4 combinaciones de `var` con 'PERPETRADOR', con proporción en hover.

    Parámetros:
    df : pd.DataFrame
        DataFrame con columnas `var`, 'VÍCTIMA' y 'PERPETRADOR'.
    var : str
        Nombre de la columna binaria.

    Retorna:
    fig : go.Figure
        Figura de Plotly con los 3 barplots.
    """
    total = len(df)
    counts = df[var].value_counts().sort_index()
    ct_vic = pd.crosstab(df[var], df['VÍCTIMA'])
    ct_per = pd.crosstab(df[var], df['PERPETRADOR'])

    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        f"Distribución de '{var}'",
        f"Combinaciones '{var}' vs VÍCTIMA",
        f"Combinaciones '{var}' vs PERPETRADOR"
    ))

    # Variable binaria
    fig.add_trace(
        go.Bar(
            x=counts.index.astype(str),
            y=counts.values,
            text=[f"{v/total:.2%}" for v in counts.values],
            textposition='auto',
            hovertemplate=f"{var}=%{{x}}<br>Conteo=%{{y}}<br>Proporción=%{{text}}<extra></extra>"
        ), row=1, col=1
    )

    # Combinaciones con VÍCTIMA
    labels_vic, ys_vic, hover_vic = [], [], []
    for i in ct_vic.index:
        for j in ct_vic.columns:
            cnt = ct_vic.at[i, j]
            prop = cnt / ct_vic.values.sum()
            labels_vic.append(f"{var}={i},VÍCTIMA={j}")
            ys_vic.append(cnt)
            hover_vic.append(f"{var}={i}<br>VÍCTIMA={j}<br>Conteo={cnt}<br>Proporción={prop:.2%}")

    fig.add_trace(
        go.Bar(x=labels_vic, y=ys_vic, hovertext=hover_vic, hoverinfo='text'),
        row=1, col=2
    )

    # Combinaciones con PERPETRADOR
    labels_per, ys_per, hover_per = [], [], []
    for i in ct_per.index:
        for j in ct_per.columns:
            cnt = ct_per.at[i, j]
            prop = cnt / ct_per.values.sum()
            labels_per.append(f"{var}={i},PERPETRADOR={j}")
            ys_per.append(cnt)
            hover_per.append(f"{var}={i}<br>PERPETRADOR={j}<br>Conteo={cnt}<br>Proporción={prop:.2%}")

    fig.add_trace(
        go.Bar(x=labels_per, y=ys_per, hovertext=hover_per, hoverinfo='text'),
        row=1, col=3
    )

    fig.update_layout(showlegend=False, width=1200, height=400,
                      margin=dict(t=50, b=50, l=50, r=50))
    return fig
def create_discretNumeric_plot_figure(df: pd.DataFrame, var: str) -> go.Figure:
    """
    Genera un objeto plotly.graph_objects.Figure de 1x3:
    1. Histograma de la variable discreta numérica `var`.
    2. Histogramas superpuestos de `var` según 'VÍCTIMA' (0 y 1).
    3. Histogramas superpuestos de `var` según 'PERPETRADOR' (0 y 1).

    Parámetros:
    df : pd.DataFrame
        DataFrame con columnas `var`, 'VÍCTIMA' y 'PERPETRADOR'.
    var : str
        Nombre de la columna discreta numérica.

    Retorna:
    fig : go.Figure
        Figura de Plotly con los 3 histogramas.
    """
    # Número de bins igual al número de valores únicos
    unique_vals = df[var].dropna().unique()
    nbins = len(unique_vals)

    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        f"Histograma de '{var}'",
        f"'{var}' vs VÍCTIMA",
        f"'{var}' vs PERPETRADOR"
    ))

    # Histograma global
    fig.add_trace(
        go.Histogram(
            x=df[var],
            nbinsx=nbins,
            hovertemplate=f"{var}=%{{x}}<br>Conteo=%{{y}}<extra></extra>"
        ),
        row=1, col=1
    )

    # Histogramas por valor de VÍCTIMA
    for val in [0, 1]:
        mask = df['VÍCTIMA'] == val
        fig.add_trace(
            go.Histogram(
                x=df.loc[mask, var],
                nbinsx=nbins,
                name=f"VÍCTIMA={val}",
                opacity=0.75,
                hovertemplate=f"{var}=%{{x}}<br>Conteo=%{{y}}<br>VÍCTIMA={val}<extra></extra>"
            ),
            row=1, col=2
        )

    # Histogramas por valor de PERPETRADOR
    for val in [0, 1]:
        mask = df['PERPETRADOR'] == val
        fig.add_trace(
            go.Histogram(
                x=df.loc[mask, var],
                nbinsx=nbins,
                name=f"PERPETRADOR={val}",
                opacity=0.75,
                hovertemplate=f"{var}=%{{x}}<br>Conteo=%{{y}}<br>PERPETRADOR={val}<extra></extra>"
            ),
            row=1, col=3
        )

    # Layout
    fig.update_layout(
        barmode='overlay',
        showlegend=True,
        width=1200,
        height=400,
        margin=dict(t=50, b=50, l=50, r=50)
    )

    return fig
def create_continuousNumeric_density_figure(df: pd.DataFrame, var: str) -> go.Figure:
    """
    Genera un objeto plotly.graph_objects.Figure de 1x3 con curvas de densidad:
    1. Densidad de la variable continua `var`.
    2. Curvas de densidad de `var` según 'VÍCTIMA' (0 y 1).
    3. Curvas de densidad de `var` según 'PERPETRADOR' (0 y 1).

    Parámetros:
    df : pd.DataFrame
        DataFrame con columnas `var`, 'VÍCTIMA' y 'PERPETRADOR'.
    var : str
        Nombre de la columna continua.

    Retorna:
    fig : go.Figure
        Figura de Plotly con los 3 density plots.
    """
    # Datos limpios
    data = df[var].dropna().values
    x_min, x_max = data.min(), data.max()
    x_grid = np.linspace(x_min, x_max, 200)

    # KDE global
    kde_all = gaussian_kde(data)
    y_all = kde_all(x_grid)

    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        f"Densidad de '{var}'",
        f"'{var}' vs VÍCTIMA",
        f"'{var}' vs PERPETRADOR"
    ))

    # Curva de densidad global
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=y_all,
            mode='lines',
            fill='tozeroy',
            name='Global',
            hovertemplate=f"{var}=%{{x:.2f}}<br>Densidad=%{{y:.3f}}<extra></extra>"
        ),
        row=1, col=1
    )

    # Densidades por valor de VÍCTIMA
    for val in [0, 1]:
        subset = df.loc[df['VÍCTIMA'] == val, var].dropna().values
        if len(subset) > 1:
            kde = gaussian_kde(subset)
            y = kde(x_grid)
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=y,
                    mode='lines',
                    fill='tonexty',
                    name=f"VÍCTIMA={val}",
                    opacity=0.75,
                    hovertemplate=f"{var}=%{{x:.2f}}<br>Densidad=%{{y:.3f}}<br>VÍCTIMA={val}<extra></extra>"
                ),
                row=1, col=2
            )
        
    # Densidades por valor de PERPETRADOR
    for val in [0, 1]:
        subset = df.loc[df['PERPETRADOR'] == val, var].dropna().values
        if len(subset) > 1:
            kde = gaussian_kde(subset)
            y = kde(x_grid)
            fig.add_trace(
                go.Scatter(
                    x=x_grid,
                    y=y,
                    mode='lines',
                    fill='tonexty',
                    name=f"PERPETRADOR={val}",
                    opacity=0.75,
                    hovertemplate=f"{var}=%{{x:.2f}}<br>Densidad=%{{y:.3f}}<br>PERPETRADOR={val}<extra></extra>"
                ),
                row=1, col=3
            )

    fig.update_layout(
        showlegend=True,
        width=1200,
        height=400,
        margin=dict(t=50, b=50, l=50, r=50)
    )

    return fig

pio.templates.default = "plotly"

def crear_figuras_umap(df):
    """
    Genera dos gráficos 3D interactivos de las componentes UMAP del DataFrame:
     1) Coloreado por 'cluster'
     2) Coloreado por 'VÍCTIMA' (0/1)
    Retorna los fragmentos HTML ya con colores embebidos.
    """
    hover_cols = list(df.columns)

    # --- Figura 1: por cluster ---
    df['cluster_str'] = df['cluster'].astype(str)
    fig1 = px.scatter_3d(
        df,
        x='umap_1', y='umap_2', z='umap_3',
        color='cluster_str',
        hover_data=hover_cols,
        title=(
            "UMAP (n_n=100, min_d=0.01, spread=0.5)<br>"
            "DBSCAN (eps=10, min_samples=3)<br>"
            "Silhouette = 0.93"
        ),
        template="plotly"
    )
    fig1.update_layout(
        legend_title_text='Cluster',
        margin=dict(l=0, r=0, t=80, b=0)
    )
    fig_html_cluster = fig1.to_html(full_html=False, include_plotlyjs='cdn')

    # --- Figura 2: por víctima ---
    df['victima_str'] = df['VÍCTIMA'].astype(str)
    fig2 = px.scatter_3d(
        df,
        x='umap_1', y='umap_2', z='umap_3',
        color='victima_str',
        hover_data=hover_cols,
        title=("Densidad de víctimas<br>"
               "por tipo de perpetrador"),
        template="plotly"
    )
    fig2.update_layout(
        legend_title_text='VÍCTIMA',
        margin=dict(l=0, r=0, t=60, b=0)
    )
    fig_html_victima = fig2.to_html(full_html=False, include_plotlyjs='cdn')

    # Limpiamos las columnas auxiliares
    df.drop(['cluster_str', 'victima_str'], axis=1, inplace=True)

    return fig_html_cluster, fig_html_victima
def plot_density_by_cluster_victima_plotly(df, var_name, html_file="density_plots.html"):
    """
    Genera un HTML con un grid 2x5 de density plots de `var_name`
    para cada cluster (0–4) y VÍCTIMA (0 en fila 1, 1 en fila 2).

    Parámetros
    ----------
    df : pandas.DataFrame
        Debe contener las columnas 'cluster', 'VÍCTIMA' y `var_name`.
    var_name : str
        Nombre de la columna numérica a plotear.
    html_file : str, opcional
        Ruta del fichero HTML de salida (por defecto "density_plots.html").

    Devuelve
    -------
    str
        Ruta al fichero HTML generado.
    """
    # Validaciones
    if var_name not in df.columns:
        raise ValueError(f"La variable '{var_name}' no está en el DataFrame.")
    for col in ('cluster','VÍCTIMA'):
        if col not in df.columns:
            raise ValueError(f"Falta la columna '{col}' en el DataFrame.")
    
    # Crear subplots
    fig = make_subplots(
        rows=2, cols=5,
        subplot_titles=[f"Cluster {c} – VÍCTIMA {v}" for v in (0,1) for c in range(5)],
        horizontal_spacing=0.03, vertical_spacing=0.1
    )
    
    # Iterar y añadir cada density plot
    for row_idx, vict in enumerate((0,1), start=1):
        for col_idx, cluster in enumerate(range(5), start=1):
            sub = df[(df['cluster']==cluster) & (df['VÍCTIMA']==vict)][var_name].dropna()
            if len(sub) >= 2:
                # calcular KDE
                kde = gaussian_kde(sub)
                xs = np.linspace(sub.min(), sub.max(), 200)
                ys = kde(xs)
                fig.add_trace(
                    go.Scatter(x=xs, y=ys, mode='lines', showlegend=False),
                    row=row_idx, col=col_idx
                )
            else:
                # si no hay datos suficientes, anotar
                fig.add_annotation(
                    x=0.5, y=0.5, text="sin datos",
                    showarrow=False,
                    xref=f"x{(row_idx-1)*5+col_idx} domain",
                    yref=f"y{(row_idx-1)*5+col_idx} domain"
                )
            # títulos de ejes
            fig.update_xaxes(title_text=var_name, row=row_idx, col=col_idx)
            fig.update_yaxes(title_text="Density", row=row_idx, col=col_idx)
    
    # Layout general
    fig.update_layout(
        title_text=f"Densidad de “{var_name}” por cluster y VÍCTIMA",
        height=600, width=2000,
        margin=dict(t=80, b=50, l=30, r=30)
    )
    
    return fig
def plot_violins_by_victima(df, var_name, html_file="violins_by_victima.html"):
    """
    Genera un HTML con dos violín plots:
     - a la izquierda: distribuciones de `var_name` en cada cluster para VÍCTIMA = 1
     - a la derecha: distribuciones de `var_name` en cada cluster para VÍCTIMA = 0

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene 'cluster', 'VÍCTIMA' y `var_name`.
    var_name : str
        Nombre de la columna numérica a representar.
    html_file : str
        Ruta de salida del HTML interactivo.
    
    Devuelve
    -------
    str
        La ruta al fichero HTML generado.
    """
    # Validaciones
    if var_name not in df.columns:
        raise ValueError(f"La variable '{var_name}' no existe en el DataFrame.")
    for col in ('cluster', 'VÍCTIMA'):
        if col not in df.columns:
            raise ValueError(f"Falta la columna '{col}' en el DataFrame.")

    # Crear subplots: 1 fila, 2 columnas
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[f"VÍCTIMA = 1", f"VÍCTIMA = 0"],
        shared_yaxes=True,
        horizontal_spacing=0.1
    )

    # Para cada valor de VÍCTIMA y cada cluster
    for col_idx, vict in enumerate([1, 0], start=1):
        for cluster in sorted(df['cluster'].unique()):
            sub = df[(df['cluster']==cluster) & (df['VÍCTIMA']==vict)][var_name].dropna()
            if len(sub) > 0:
                fig.add_trace(
                    go.Violin(
                        x=[str(cluster)] * len(sub),
                        y=sub,
                        name=f"Cluster {cluster}",
                        showlegend=(col_idx == 1),  # leyenda solo en el primer gráfico
                        spanmode='hard'
                    ),
                    row=1, col=col_idx
                )
            else:
                # Si no hay datos, lo anotamos
                fig.add_annotation(
                    x=0.5, y=0.5, text="sin datos",
                    xref=f"x{(col_idx-1)+1} domain",
                    yref=f"y{(col_idx-1)+1} domain",
                    showarrow=False,
                    row=1, col=col_idx
                )
        # Ejes
        fig.update_xaxes(title_text="Cluster", row=1, col=col_idx)
    fig.update_yaxes(title_text=var_name, row=1, col=1)

    # Layout general
    fig.update_layout(
        title_text=f"Violin plots de “{var_name}” por cluster y VÍCTIMA",
        width=1200, height=500,
        margin=dict(t=80, l=60, r=60, b=60)
    )

    # Guardar y devolver ruta
    #fig.write_html(html_file, include_plotlyjs='cdn')
    return fig
def plot_violins_by_cluster(df, var_name, html_file=None):
    """
    Genera un violin plot de `var_name` para cada cluster.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que debe contener las columnas 'cluster' y `var_name`.
    var_name : str
        Nombre de la columna numérica a representar.
    html_file : str o None
        Si se proporciona ruta, guarda el gráfico interactivo como HTML.
        Si es None, no guarda fichero y solo devuelve la figura.

    Devuelve
    -------
    fig : plotly.graph_objects.Figure
        El objeto Figure con los violines por cluster.
    """
    # Validaciones
    if var_name not in df.columns:
        raise ValueError(f"La variable '{var_name}' no existe en el DataFrame.")
    if 'cluster' not in df.columns:
        raise ValueError("Falta la columna 'cluster' en el DataFrame.")

    # Crear la figura
    fig = go.Figure()

    # Por cada cluster, añadir su violin
    for cluster in sorted(df['cluster'].unique()):
        sub = df[df['cluster'] == cluster][var_name].dropna()
        if len(sub) > 0:
            fig.add_trace(
                go.Violin(
                    x=[str(cluster)] * len(sub),
                    y=sub,
                    name=f"Cluster {cluster}",
                    spanmode='hard',
                )
            )
        else:
            # Si no hay datos, lo anotamos en el centro del cluster
            fig.add_annotation(
                x=str(cluster), y=0,
                text="sin datos",
                showarrow=False,
                yanchor="bottom"
            )

    # Ejes y layout
    fig.update_xaxes(title_text="Cluster")
    fig.update_yaxes(title_text=var_name)
    fig.update_layout(
        title_text=f"Distribución de “{var_name}” por cluster",
        violinmode='group',
        width=800, height=500,
        margin=dict(t=80, l=60, r=60, b=60)
    )

    # Guardar HTML si se solicitó
    if html_file:
        fig.write_html(html_file, include_plotlyjs='cdn')

    return fig
def plot_umap_cluster_distribution(df, cluster_id):
    """
    Genera una figura Plotly 1x3 con la distribución (histograma de densidad)
    de las columnas umap_1, umap_2, umap_3 para un cluster específico.

    Parámetros
    ----------
    df : pandas.DataFrame
        Debe contener al menos las columnas:
        - 'cluster'
        - 'umap_1', 'umap_2', 'umap_3'
    cluster_id : int
        Valor del cluster (por ejemplo 0,1,2,3 o 4) que queremos visualizar.

    Devuelve
    -------
    fig : plotly.graph_objects.Figure
        Figura con subplots 1x3 para el cluster indicado.
    """
    # Filtrar el DataFrame por el cluster deseado
    df_c = df[df['cluster'] == cluster_id]
    if df_c.empty:
        raise ValueError(f"No hay filas para el cluster {cluster_id}")

    umap_cols = ['umap_1', 'umap_2', 'umap_3']

    # Crear subplots 1x3
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[f"Cluster {cluster_id} · {col}" for col in umap_cols],
        horizontal_spacing=0.1
    )

    # Añadir histogramas normalizados a densidad
    for j, col in enumerate(umap_cols):
        fig.add_trace(
            go.Histogram(
                x=df_c[col],
                histnorm='probability density',
                showlegend=False
            ),
            row=1, col=j+1
        )

    # Ajustes de layout
    fig.update_layout(
        title_text=f"Distribución UMAP del Cluster {cluster_id}",
        height=400,  # altura total
        width=1200,  # ancho total
        bargap=0.1
    )

    # Etiquetas comunes
    fig.update_xaxes(title_text="Valor UMAP")
    fig.update_yaxes(title_text="Densidad")

    return fig
def plot_victimas_por_cluster(
        df: pd.DataFrame,
        binary_var: str,
        cluster_col: str = "cluster",
        victim_col: str = "VÍCTIMA"
    ):
    """
    Crea un gráfico 2×5 donde:
      * Fila 1 → proporción (No-víctima / Víctima) dentro de cada cluster
                  para el primer valor de `binary_var` (p.ej. 0).
      * Fila 2 → proporción (No-víctima / Víctima) dentro de cada cluster
                  para el segundo valor de `binary_var` (p.ej. 1).

    Parameters
    ----------
    df : pd.DataFrame
        Datos originales.
    binary_var : str
        Nombre de la columna binaria sobre la que segmentar.
    cluster_col : str, default="cluster"
        Nombre de la columna de clusters.
    victim_col : str, default="VÍCTIMA"
        Nombre de la columna binaria que indica víctima (1) / no víctima (0).

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # 1. Comprobaciones básicas
    valores_bin = sorted(df[binary_var].dropna().unique())
    if len(valores_bin) != 2:
        raise ValueError(
            f"La columna {binary_var} debe tener exactamente dos valores distintos."
        )

    clusters = sorted(df[cluster_col].dropna().unique())
    if len(clusters) > 5:                      # Nos quedamos con los cinco primeros
        clusters = clusters[:5]

    # 2. Crear la figura vacía
    fig = make_subplots(
        rows=2, cols=len(clusters),
        shared_yaxes=True,
        subplot_titles=[f"Cluster {c}" for c in clusters] * 2,
        vertical_spacing=0.12,
        horizontal_spacing=0.04
    )

    # 3. Añadir trazas
    colores = {"No víctima": "#636EFA", "Víctima": "#EF553B"}  # opcional
    showlegend_done = {"No víctima": False, "Víctima": False}

    for col_idx, cl in enumerate(clusters, start=1):
        for row_idx, bin_val in enumerate(valores_bin, start=1):
            sub = df[(df[cluster_col] == cl) & (df[binary_var] == bin_val)]
            prop = (
                sub[victim_col]
                .value_counts(normalize=True)
                .reindex([0, 1])           # Asegura orden 0, 1
                .fillna(0)
            )

            for vict, etiqueta in zip([0, 1], ["No víctima", "Víctima"]):
                fig.add_trace(
                    go.Bar(
                        x=[etiqueta],
                        y=[prop[vict]],
                        name=etiqueta,
                        marker_color=colores[etiqueta],
                        showlegend=not showlegend_done[etiqueta]
                    ),
                    row=row_idx,
                    col=col_idx
                )
                showlegend_done[etiqueta] = True

    # 4. Ajustes finales
    fig.update_layout(
        height=650,
        width=1300,
        barmode="stack",
        title_text=(
            f"Proporción Víctima / No-víctima por {binary_var} y cluster"
            f"<br>(fila 1: {binary_var} = {valores_bin[0]}, "
            f"fila 2: {binary_var} = {valores_bin[1]})"
        ),
        legend_title_text="Estado"
    )
    fig.update_yaxes(tickformat=".0%")

    return fig
def plot_importancias_umap_norm(df_imp: pd.DataFrame) -> go.Figure:
    """
    Barplots normalizados (0-1) con texto de la variable en cada barra.
    
    Parameters
    ----------
    df_imp : pd.DataFrame
        Índice = variables.  Columnas = 'umap_1', 'umap_2', 'umap_3'.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    # --- 1) Comprobaciones mínimas
    comps = ["umap_1", "umap_2", "umap_3"]
    if not all(c in df_imp.columns for c in comps):
        raise ValueError(f"El DataFrame debe contener las columnas {comps}")

    # --- 2) Normalización columna a columna (0-1)
    df_norm = df_imp[comps].div(df_imp[comps].max())

    # --- 3) Orden de variables (mayor importancia en umap_1 primero)
    orden_vars = df_norm["umap_1"].sort_values(ascending=False).index

    # --- 4) Figura 1×3
    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.08,
        subplot_titles=("UMAP 1 (normalizado)", "UMAP 2 (normalizado)", "UMAP 3 (normalizado)")
    )

    for i, comp in enumerate(comps, start=1):
        # Datos ordenados
        vals = df_norm.loc[orden_vars, comp]

        fig.add_trace(
            go.Bar(
                x=vals,
                y=orden_vars,
                orientation="h",
                #text=orden_vars,           # nombre de la variable dentro de la barra
                #textposition="inside",
                insidetextanchor="middle",
                hovertemplate="<b>%{y}</b><br>Importancia: %{x:.2f}<extra></extra>",
                name=comp.upper()
            ),
            row=1, col=i
        )

    # --- 5) Ajustes globales
    fig.update_layout(
        height=650,
        width=1400,
        showlegend=False,
        title_text="Importancia normalizada (0-1) de las variables por componente UMAP",
        margin=dict(l=120, r=40, t=90, b=40)
    )
    fig.update_xaxes(title_text="Importancia (0-1)", range=[0, 1])
    fig.update_yaxes(title_text="Variable", autorange="reversed")  # la más importante arriba

    return fig

#EDA plots-------------------------------------------------------------------------------------------------
def plot_binary_proportion(df: pd.DataFrame, column_name: str):
    """
    Plots an interactive proportion bar chart for a binary column using Plotly,
    with a discrete legend (no gradient color bar).
    """
    # 1) Validations
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame.")
    if df[column_name].nunique() != 2:
        raise ValueError(f"Column '{column_name}' is not binary.")
    
    # 2) Compute proportions and explicitly set column names
    proportions = (
        df[column_name]
        .value_counts(normalize=True)
        .reset_index()
    )
    # Now we know reset_index() created two columns: "index" and the original series name.
    proportions.columns = ['Category', 'Proportion']    # ← here’s the fix
    
    # 3) Ensure categories are treated as discrete
    proportions['Category'] = proportions['Category'].astype(str)
    
    # 4) Map your two colors
    color_map = {
        proportions.loc[0, 'Category']: '#1f77b4',
        proportions.loc[1, 'Category']: '#ff7f0e',
    }
    
    # 5) Build the bar chart
    fig = px.bar(
        proportions,
        x='Category',
        y='Proportion',
        text='Proportion',
        color='Category',
        color_discrete_map=color_map,
        title=f'Proportion of "{column_name}"',
        labels={'Proportion': 'Proportion (%)'},
        height=500,
    )
    
    # 6) Tweak traces & layout
    fig.update_traces(
        texttemplate='%{text:.1%}',
        hovertemplate='<b>%{x}</b><br>Proportion: %{y:.1%}<extra></extra>',
        marker_line_width=0
    )
    fig.update_layout(
        coloraxis_showscale=False,    # make sure no colorbar appears
        yaxis_tickformat='.0%',
        yaxis_range=[0, 1],
        showlegend=True,
        legend_title_text='Category',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        hovermode='x'
    )
    
    return fig
def plot_three_binary_proportions(df: pd.DataFrame) -> go.Figure:
    """
    Plots a 1×3 grid of proportion bar charts for three columns:
    "VÍCTIMA", "PERPETRADOR", and "VICTIMA_PERPETRADOR".  
    - If a column has exactly two categories, it plots both.  
    - If a column has exactly one category, it plots a single bar.  
    - If a column has more than two categories, it raises an error.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame; must contain the three specified columns.

    Returns
    -------
    go.Figure
        A figure with 1 row and 3 bar‐chart subplots.

    Raises
    ------
    KeyError
        If any of the required columns is missing.
    ValueError
        If any column has more than two unique values.
    """
    cols = ["VÍCTIMA", "PERPETRADOR", "VICTIMA_PERPETRADOR"]
    # 1) Validate presence and cardinality
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame.")
        unique_count = df[col].nunique(dropna=False)
        if unique_count > 2:
            raise ValueError(f"Column '{col}' has {unique_count} categories; expected at most 2.")

    # 2) Create 1×3 subplots, sharing the y-axis
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=cols,
        shared_yaxes=True
    )

    # 3) Build each subplot
    default_colors = ["#1f77b4", "#ff7f0e"]
    for i, col in enumerate(cols, start=1):
        counts = df[col].value_counts(normalize=True, dropna=False)
        categories = counts.index.astype(str).tolist()
        proportions = counts.values.tolist()

        # assign colors in order, repeating if needed
        color_map = {
            cat: default_colors[idx % len(default_colors)]
            for idx, cat in enumerate(categories)
        }

        # add one bar per category
        for cat, prop in zip(categories, proportions):
            fig.add_trace(
                go.Bar(
                    x=[cat],
                    y=[prop],
                    name=cat,
                    legendgroup=cat,
                    showlegend=(i == 1),       # only show legend once
                    marker_color=color_map[cat],
                    text=[prop],
                    texttemplate="%{text:.1%}",
                    hovertemplate="<b>%{x}</b><br>Proportion: %{y:.1%}<extra></extra>",
                    marker_line_width=0
                ),
                row=1, col=i
            )

        # y‐axis formatting for each subplot
        fig.update_yaxes(
            range=[0, 1],
            tickformat=".0%",
            row=1, col=i
        )

    # 4) Global layout tweaks
    fig.update_layout(
        title_text="Proportion of Categories in VÍCTIMA / PERPETRADOR / VICTIMA_PERPETRADOR",
        height=500,
        plot_bgcolor="white",
        hovermode="x unified",
        legend_title_text="Category",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        )
    )

    return fig
def plot_categorical_proportions(df: pd.DataFrame, column_name: str):
    """
    Plots an interactive proportion bar chart for a categorical or ordinal column using Plotly,
    with a discrete legend (no gradient color bar).

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    column_name : str
        Name of the categorical / ordinal column.

    Returns
    -------
    plotly.graph_objs._figure.Figure
        Interactive bar plot of proportions.

    Raises
    ------
    KeyError
        If column_name is not in df.
    ValueError
        If column_name has fewer than 1 unique value.
    """
    # 1) Validations
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in DataFrame.")
    unique_count = df[column_name].nunique(dropna=False)
    if unique_count < 1:
        raise ValueError(f"Column '{column_name}' has no values to plot.")
    
    # 2) Compute proportions (excluding NaNs)
    proportions = (
        df[column_name]
        .dropna()
        .value_counts(normalize=True)
        .reset_index()
    )
    proportions.columns = ['Category', 'Proportion']
    
    # 3) Ensure categories are treated as discrete
    proportions['Category'] = proportions['Category'].astype(str)
    
    # 4) Build the bar chart
    fig = px.bar(
        proportions,
        x='Category',
        y='Proportion',
        text='Proportion',
        color='Category',
        title=f'Proportion of "{column_name}"',
        labels={'Proportion': 'Proportion (%)'},
        height=500,
    )
    
    # 5) Tweak traces & layout
    fig.update_traces(
        texttemplate='%{text:.1%}',
        hovertemplate='<b>%{x}</b><br>Proportion: %{y:.1%}<extra></extra>',
        marker_line_width=0
    )
    fig.update_layout(
        coloraxis_showscale=False,
        yaxis_tickformat='.0%',
        yaxis_range=[0, 1],
        showlegend=True,
        legend_title_text='Category',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor='white',
        hovermode='x'
    )
    
    return fig
def plot_cont_distribution(df: pd.DataFrame, var: str) -> go.Figure:
    """
    Plots a single density curve for the continuous column `var` using Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the variable.
    var : str
        Name of the continuous column to plot.

    Returns
    -------
    go.Figure
        A Plotly Figure with one density plot.

    Raises
    ------
    KeyError
        If `var` is not in `df`.
    ValueError
        If `var` has fewer than 2 non-null values.
    """
    # 1) Validate
    if var not in df.columns:
        raise KeyError(f"Column '{var}' not found in DataFrame.")
    data = df[var].dropna().values
    if data.size < 2:
        raise ValueError(f"Column '{var}' must have at least 2 non-null values for a density plot.")

    # 2) Define grid over which to evaluate KDE
    x_min, x_max = data.min(), data.max()
    x_grid = np.linspace(x_min, x_max, 200)

    # 3) Compute KDE
    kde = gaussian_kde(data)
    y = kde(x_grid)

    # 4) Build figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=y,
            mode='lines',
            fill='tozeroy',
            name='Density',
            hovertemplate=f"{var}=%{{x:.2f}}<br>Density=%{{y:.3f}}<extra></extra>"
        )
    )

    # 5) Layout tweaks
    fig.update_layout(
        title=f"Density Plot of '{var}'",
        xaxis_title=var,
        yaxis_title="Density",
        plot_bgcolor="white",
        hovermode="x",
        showlegend=False
    )

    return fig

#QUIZZ plots---------------------------------------------------------------------------------------------
def shap_vizz(shap_dict: Dict[str, Tuple[float, int]]) -> go.Figure:
    """
    Create a horizontal SHAP importance bar chart.
    
    Parameters
    ----------
    shap_dict : Dict[str, Tuple[float, int]]
        Mapping feature → (percent_importance, sign_flag), where
        sign_flag is +1 or -1.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        A horizontal bar chart with negative bars in red (left) and
        positive bars in blue (right).
    """
    # Prepare data: signed values and colors
    # Sort by signed value so negatives appear at top going left, positives at bottom.
    items = sorted(
        shap_dict.items(),
        key=lambda kv: kv[1][0] * kv[1][1]
    )
    features = []
    values = []
    colors = []
    for feat, (pct, sign) in items:
        features.append(feat)
        values.append(pct * sign)
        colors.append('blue' if sign == 1 else 'red')

    # Build the figure
    fig = go.Figure(
        go.Bar(
            x=values,
            y=features,
            orientation='h',
            marker_color=colors,
            hovertemplate='%{y}: %{x:.2f}%%<extra></extra>'
        )
    )
    
    # Style layout
    fig.update_layout(
        xaxis_title='SHAP Importance (%)',
        yaxis_title='Feature',
        template='simple_white',
        bargap=0.3,
        margin=dict(l=150, r=50, t=50, b=50)
    )
    
    return fig

