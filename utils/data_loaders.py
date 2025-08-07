import streamlit as st 
import pandas as pd

@st.cache_data
def load_csv_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

@st.cache_data
def load_excel_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

@st.cache_data
def load_csv_target_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path).fillna(0)

@st.cache_data
def load_special_csv_data(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=';',
        low_memory=False
    ).drop(columns="ID_0")