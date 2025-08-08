import streamlit as st
from st_pages import get_nav_from_toml, add_page_title
import base64


def add_logo_only(png_path: str, height_px: int = 100):
    # 1) Embed the local file as Base64
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    # 2) Inject CSS that *only* sets the sidebar-nav background image
    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: url("data:image/png;base64,{b64}");
                background-repeat: no-repeat;
                background-position: center top;
                padding-top: {height_px}px;  /* push links below the logo */
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# 1. Page config
st.set_page_config(
    page_title="Crime Data Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Load your pages TOML
nav = get_nav_from_toml(".streamlit/pages.toml")

# 3. Sidebar: inject only the logo, then navigation
with st.sidebar:
    add_logo_only("img/logo.png", height_px=100)
    pg = st.navigation(nav)

# 4. Title + run
add_page_title(pg)
pg.run()
