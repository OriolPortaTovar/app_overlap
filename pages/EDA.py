import streamlit as st

progress = st.progress(0, text="Opening aplication…")


progress.progress(20, text="Initializing Icrea App…")
import pandas as pd
from utils.data_objects import (
    taxonomic_dict,
    lista_global_vars,
    lista_perpetrador,
    lista_victima,
    target_col,
    min_vars, 
    information
)

split_list = lista_global_vars + lista_perpetrador + lista_victima + target_col
keys_list = list(taxonomic_dict.keys())

progress.progress(40, text="Loading data loaders…")
from utils.data_loaders import (
    load_special_csv_data
)

progress.progress(60, text="Loading data processors…")
from utils.data_processor import (
    eda_processing_pipeline,
    get_predictions,
    get_unique_values,
    missing_summary,
    filter_by_values,
    get_range_list
)

progress.progress(80, text="Loading data visualizers…")
from utils.data_visualization import (
    plot_binary_proportion,
    plot_three_binary_proportions,
    plot_categorical_proportions,
    plot_cont_distribution
)


progress.progress(90, text="Enabling tools..")
st.set_page_config(
    page_title="Exploratory Data Analysis", 
    layout="wide",
    menu_items={
        "About": information,
        "Get Help": "mailto:oriolportamst@gmail.com",
        "Report a Bug": " https://github.com/legna29A/app_overlap.com"
    }
)

with st.sidebar:
    uploaded_file = st.file_uploader(
        "Choose an CSV file",
        type=["csv"],
        accept_multiple_files=False
    )
    if uploaded_file is None:
        st.image("img/arrow.gif", use_container_width = True)
    progress.progress(100, text="File uploaded – processing now…")
    progress.empty()

    
col_raw, col_transform = st.columns(2)
if uploaded_file:
    
    with col_raw:
        st.header("Raw Data")
        raw_df = load_special_csv_data(uploaded_file)
        raw_df = raw_df[min_vars]
        st.dataframe(raw_df)
    
    with col_transform:
        st.header("Processed Data")
        with st.spinner("Processing…"):
            processed = eda_processing_pipeline(raw_df, split_list)
            f_df = get_predictions(processed, lista_global_vars)
            st.dataframe(f_df)
    
    
    if len(f_df)>0: 
        st.header("Deep Dive")
        tabraw,tabprocessed = st.tabs(["🧮 Raw Data", "🎨 Processed Data"])
        
        with tabraw:
            st.subheader("Analyse Raw Variables")
            colvar1,colplots1 = st.columns([1,3])
            colvar2,colplots2 = st.columns([1,3])
            df_filtered_val = pd.DataFrame()
            df_filtered_values = pd.DataFrame()
            with colvar1:
                column = st.selectbox(
                    label  = "Select a variable:",
                    options = ["--"] + raw_df.columns.tolist(),
                )
                if column != "--":
                    missing_report = missing_summary(raw_df, column)
                    st.markdown(missing_report)
                    if column in keys_list:
                        description = taxonomic_dict[column][1]
                        st.write(description)
                    col_values = get_unique_values(raw_df,column)
            with colplots1:
                if column != "--":
                    if len(col_values) == 2:
                        bi_fig = plot_binary_proportion(raw_df,column)
                        st.plotly_chart(bi_fig, use_container_width=True)
                    if len(col_values) > 2 and  len(col_values) < 11:
                        cat_fig = plot_categorical_proportions(raw_df,column)
                        st.plotly_chart(cat_fig, use_container_width=True)
                    if len(col_values)>= 11:
                        cont_fig = plot_cont_distribution(raw_df,column)
                        st.plotly_chart(cont_fig, use_container_width=True)                 
            with colvar2:
                if column != "--":
                    if len(col_values) == 2:
                        st.subheader("Analyse with Target")
                        bi_radio = st.radio(
                            "Select a value to check",
                            [col_values[0], col_values[1]],
                            index=None, 
                        )
                        if bi_radio is not None:
                            df_filtered = filter_by_values(raw_df,column,[bi_radio])
                    if len(col_values) > 2 and  len(col_values) < 11:
                        st.subheader("Analyse with Target")
                        cat_var = st.multiselect(
                            "Select the categories to filter",
                            col_values,
                        ) 
                        if cat_var is not None:
                            df_filtered = filter_by_values(raw_df,column,cat_var)
                    if len(col_values)>= 11:
                        st.subheader("Analyse with Target")
                        value = st.select_slider(
                            "Select a single value",
                            options=col_values,
                        )
                        start_val, end_val = st.select_slider(
                            "Select a range of values",
                            options=col_values,
                            value=(col_values[0],col_values[0]),
                        )
                        cccol1,cccol2 = st.columns([1,0.8])
                        with cccol1:
                            pass
                        with cccol2:
                            clicked = st.button("🔍 Filter")
                        if value is not None and clicked:
                            df_filtered_val = filter_by_values(raw_df,column,[value])
                        if start_val != end_val and clicked:
                            range = get_range_list(col_values,start_val, end_val)
                            df_filtered_values = filter_by_values(raw_df,column,range)              
            with colplots2:
                if column != "--":
                    if len(col_values) == 2:
                        col_target1,col_target_3 = st.columns([1,3])
                        if bi_radio is not None and len(df_filtered)>0:
                            with st.spinner("Loading binary filter…"):
                                with col_target1:
                                    st.write(f"**Filtered Data:**  {column} {bi_radio}")
                                    st.dataframe(df_filtered)
                                with col_target_3:
                                    fig_tree_bi=plot_three_binary_proportions(df_filtered)
                                    st.plotly_chart(fig_tree_bi, use_container_width=True)
                    if len(col_values) > 2 and  len(col_values) < 11:
                        col_target1,col_target_3 = st.columns([1,3])
                        if cat_var is not None and len(df_filtered)>0:
                            with st.spinner("Loading categorical filter…"):
                                with col_target1:
                                    st.write(f"**Filtered Data:**  {column} {cat_var}")
                                    st.dataframe(df_filtered)
                                with col_target_3:
                                    fig_tree_bi=plot_three_binary_proportions(df_filtered)
                                    st.plotly_chart(fig_tree_bi, use_container_width=True)
                    if len(col_values)>= 11:
                        col_target1,col_target_3 = st.columns([1,3])
                        if value is not None and len(df_filtered_val)>0 and clicked:
                            with st.spinner("Loading value filter…"):
                                with col_target1:
                                    st.write(f"**Filtered Data:**  {column}: {value}")
                                    st.dataframe(df_filtered_val)
                                with col_target_3:
                                    fig_tree_cont=plot_three_binary_proportions(df_filtered_val)
                                    st.plotly_chart(fig_tree_cont, use_container_width=True)
                        if start_val != end_val and len(df_filtered_values)>0 and clicked:
                            with st.spinner("Loading range filter…"):
                                with col_target1:
                                    st.write(f"**Filtered Data:**  {column}: [{start_val},{end_val}]")
                                    st.dataframe(df_filtered_values)
                                with col_target_3:
                                    fig_tree_cont_v=plot_three_binary_proportions(df_filtered_values)
                                    st.plotly_chart(fig_tree_cont_v, use_container_width=True)                    

        with tabprocessed:
            st.subheader("Analyse Processed Variables")
            colvar1,colplots1 = st.columns([1,3])
            colvar2,colplots2 = st.columns([1,3])
            df_filtered_val = pd.DataFrame()
            df_filtered_values = pd.DataFrame()
            with colvar1:
                column = st.selectbox(
                    label  = "Select a variable:",
                    options = ["--"] + f_df.columns.tolist(),
                )
                if column != "--":
                    missing_report = missing_summary(f_df, column)
                    st.markdown(missing_report)
                    if column in keys_list:
                        description = taxonomic_dict[column][1]
                        st.write(description)
                    col_values = get_unique_values(f_df,column)
            with colplots1:
                if column != "--":
                    if len(col_values) == 2:
                        bi_fig = plot_binary_proportion(f_df,column)
                        st.plotly_chart(bi_fig, use_container_width=True)
                    if len(col_values) > 2 and  len(col_values) < 11:
                        cat_fig = plot_categorical_proportions(f_df,column)
                        st.plotly_chart(cat_fig, use_container_width=True)
                    if len(col_values)>= 11:
                        cont_fig = plot_cont_distribution(f_df,column)
                        st.plotly_chart(cont_fig, use_container_width=True)                 
            with colvar2:
                if column != "--":
                    if len(col_values) == 2:
                        st.subheader("Analyse with Target")
                        bi_radio = st.radio(
                            "Select a value to check",
                            [col_values[0], col_values[1]],
                            index=None, 
                        )
                        if bi_radio is not None:
                            df_filtered = filter_by_values(f_df,column,[bi_radio])
                    if len(col_values) > 2 and  len(col_values) < 11:
                        st.subheader("Analyse with Target")
                        cat_var = st.multiselect(
                            "Select the categories to filter",
                            col_values,
                        ) 
                        if cat_var is not None:
                            df_filtered = filter_by_values(f_df,column,cat_var)
                    if len(col_values)>= 11:
                        st.subheader("Analyse with Target")
                        value = st.select_slider(
                            "Select a single value",
                            options=col_values,
                        )
                        start_val, end_val = st.select_slider(
                            "Select a range of values",
                            options=col_values,
                            value=(col_values[0],col_values[0]),
                        )
                        cccol1,cccol2 = st.columns([1,0.8])
                        with cccol1:
                            pass
                        with cccol2:
                            clicked = st.button("🔍 Filter")
                        if value is not None and clicked:
                            df_filtered_val = filter_by_values(f_df,column,[value])
                        if start_val != end_val and clicked:
                            range = get_range_list(col_values,start_val, end_val)
                            df_filtered_values = filter_by_values(f_df,column,range)                          
            with colplots2:
                if column != "--":
                    if len(col_values) == 2:
                        col_target1,col_target_3 = st.columns([1,3])
                        if bi_radio is not None and len(df_filtered)>0:
                            with st.spinner("Loading binary filter…"):
                                with col_target1:
                                    st.write(f"**Filtered Data:**  {column} {bi_radio}")
                                    st.dataframe(df_filtered)
                                with col_target_3:
                                    fig_tree_bi=plot_three_binary_proportions(df_filtered)
                                    st.plotly_chart(fig_tree_bi, use_container_width=True)
                    if len(col_values) > 2 and len(col_values) < 11:
                        col_target1,col_target_3 = st.columns([1,3])
                        if cat_var is not None and len(df_filtered)>0:
                            with st.spinner("Loading categorical filter…"):
                                with col_target1:
                                    st.write(f"**Filtered Data:**  {column} {cat_var}")
                                    st.dataframe(df_filtered)
                                with col_target_3:
                                    fig_tree_bi=plot_three_binary_proportions(df_filtered)
                                    st.plotly_chart(fig_tree_bi, use_container_width=True)
                    if len(col_values)>= 11:
                        col_target1,col_target_3 = st.columns([1,3])
                        if value is not None and len(df_filtered_val)>0 and clicked:
                            with st.spinner("Loading value filter…"):
                                with col_target1:
                                    st.write(f"**Filtered Data:**  {column}: {value}")
                                    st.dataframe(df_filtered_val)
                                with col_target_3:
                                    fig_tree_cont=plot_three_binary_proportions(df_filtered_val)
                                    st.plotly_chart(fig_tree_cont, use_container_width=True)
                        if start_val != end_val and len(df_filtered_values)>0 and clicked:
                            with st.spinner("Loading range filter…"):
                                with col_target1:
                                    st.write(f"**Filtered Data:**  {column}: [{start_val},{end_val}]")
                                    st.dataframe(df_filtered_values)
                                with col_target_3:
                                    fig_tree_cont_v=plot_three_binary_proportions(df_filtered_values)
                                    st.plotly_chart(fig_tree_cont_v, use_container_width=True)                    


