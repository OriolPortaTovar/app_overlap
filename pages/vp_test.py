import streamlit as st

progress = st.progress(0, text="Opening Quiz…")
import json

progress.progress(20, text="Initializing Quiz...")
from utils.data_objects import(
    EUROPEAN_COUNTRIES,
    AFRICAN_COUNTRIES,
    AMERICAN_COUNTRIES,
    ASIAN_COUNTRIES,
    OCEANIA_COUNTRIES,
    ETHNIC_GROUPS,
    SEXUAL_ORIENTATIONS,
    dict_vars,
    information_schema
)

progress.progress(40, text="Loading data loaders…")
from utils.data_loaders import (
    load_excel_data,
    load_csv_data
)
progress.progress(50, text="Loading processors…")
from utils.data_processor import (
    get_model_vars,
    transform_json_to_pca,
    classify_dt_pcs,
    classify_pcs_nn,
    update_dict_vars,
    infer_pc_shap_dt,
    infer_pc_shap_nn,
    shap_original_from_pcs,
    shap_to_percent_with_sign,
    json2md_variables_formater,
    metric_parser,
    combine_shap_percent,
    rebuild_shap_dict,
    overlap_metric_parser,
    _p
)


progress.progress(60, text="Loading Predictors…")
from utils.data_visualization import(
    shap_vizz
)
from pathlib import Path
APP_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = APP_ROOT / "data"
feature_pc_weights_path         = _p(DATA_DIR / "NN"  / "nn_model.h5")
#feature_pc_weights_path = r"data\feat_vs_pc_importance.csv"


st.set_page_config(page_title="Predictive Quiz",layout="wide",
    menu_items={
        "About": """
            ### ℹ️ Predictive Quiz

            This application lets you:

            * Explore the training data (EDA tab).
            * Upload or fill out a survey to get victim/perpetrator risk predictions (Predictive Quiz tab).
            * Download every table, figure and result.

            Need more help? Reach us at support@example.com
        """,
        "Get Help": "mailto:oriolportamst@gmail.com",
        "Report a Bug": " https://github.com/legna29A/app_overlap.com"
    }
)


with st.sidebar:
    uploaded_file = st.file_uploader(
        "Choose an Excel file",
        type=["xlsx", "xls"],
        accept_multiple_files=False
    )
    if uploaded_file is None:
        col1,col2,col3 = st.columns(3)
        with col1: 
            pass
        with col2: 
            pass
        with col3:
            with st.popover("ℹ️"):
                st.markdown(information_schema)
    progress.progress(100, text="Start Quizz")
    progress.empty()
    
if uploaded_file is not None:
    feat_pc_importance_df = load_csv_data(feature_pc_weights_path).drop(columns="Unnamed: 0")
    # 1) Load raw data and extract model variables
    df = load_excel_data(uploaded_file)
    df_final = get_model_vars(df)
    data_json = df_final.to_dict(orient="records")

    # 2) Transform to PCA (returns JSON string) and parse
    pca_json_str = transform_json_to_pca(data_json)
    pca_list = json.loads(pca_json_str)

    # 3) Run classifiers (each returns JSON string) and parse
    vict_json_str = classify_dt_pcs(pca_list)
    perp_json_str = classify_pcs_nn(pca_list)
    vict = json.loads(vict_json_str)[0] if vict_json_str else {}
    perp = json.loads(perp_json_str)[0] if perp_json_str else {}

    # 4) Slice PC dicts for SHAP
    pcs_full = pca_list[0]
    pc_dict_dt = {f"PC{i}": pcs_full[f"PC{i}"] for i in range(1, 19)}
    pc_dict_nn = {f"PC{i}": pcs_full[f"PC{i}"] for i in range(1, 23)}

    # 5) Compute SHAP values
    shap_dt = infer_pc_shap_dt(pc_dict_dt)
    shap_nn = infer_pc_shap_nn(pc_dict_nn)

    shap_dt_by_feature = shap_original_from_pcs(feat_pc_importance_df, shap_dt)
    shap_nn_by_feature = shap_original_from_pcs(feat_pc_importance_df, shap_nn)
    
    shap_dt_by_feature = shap_to_percent_with_sign(shap_dt_by_feature)
    shap_nn_by_feature = shap_to_percent_with_sign(shap_nn_by_feature)
    # --- Combined results JSON (what you want to show + download) ---
    
    comb_shap = combine_shap_percent(shap_dt_by_feature,shap_nn_by_feature)
    comb_shap_filtered = { key: val_sign[0] for key, val_sign in comb_shap.items() }
    comb_shap_scaled = shap_to_percent_with_sign(comb_shap_filtered)
    overlap_shap = rebuild_shap_dict(comb_shap_scaled,comb_shap)
    
    prediction_results = {
        "Decision Tree": vict,
        "Neural Network": perp,
        "SHAP_DT_pcs": shap_dt,
        "SHAP_NN_pcs": shap_nn,
        "SHAP_DT_vars": shap_dt_by_feature,
        "SHAP_NN_vars": shap_nn_by_feature,
        "SHAP_overlap": overlap_shap
    }
    # Model vars: extract single dict and preserve accents
    model_vars_obj = data_json[0]
    model_vars_str = json.dumps(model_vars_obj, indent=2, ensure_ascii=False)
    # PCA: extract single dict and pretty-print
    pca_obj = pca_list[0]
    pca_str = json.dumps(pca_obj, indent=2, ensure_ascii=False)
    # Sidebar: download buttons

    with st.sidebar:
        st.download_button(
            label="Download Vars JSON",
            data=model_vars_str,
            file_name="model_vars.json",
            mime="application/json",
            use_container_width=True,
            type = 'tertiary'
        )
        st.download_button(
            label="Download PCA JSON",
            data=pca_str,
            file_name="pca_data.json",
            mime="application/json",
            use_container_width=True,
            type = 'tertiary'
        )
        st.download_button(
            label="Download Results",
            data=json.dumps(prediction_results, indent=2, ensure_ascii=False),
            file_name="prediction_results.json",
            mime="application/json",
            use_container_width=True,
            type = 'primary'
        )

    tab_user,tab_admin = st.tabs(["🚀 User Insights","🛠️ Administrator Insights"])
    
    with tab_user:
        desc_col,graph_col_dt,graph_col_nn = st.columns([1,1,1])
        with desc_col:
            st.header("📝 Subject Summary")
            with st.container(border=True,height=600):
                DEFAULT_SECTION_TEMPLATE = (
                    "{description}\n\n"
                    "**Assigned value:** {value}"
                )
                var_report = json2md_variables_formater(DEFAULT_SECTION_TEMPLATE,model_vars_obj)
                st.markdown(var_report)
        with graph_col_dt:
            st.header("🌳 Victim Results")
            delta_color_dt,value_dt,delta_dt = metric_parser(vict)
            coldt1, coldt2, coldt3 = st.columns(3)
            coldt2.metric(label="Victim Risk", value=value_dt, delta=delta_dt, delta_color=delta_color_dt)
            fig_dt = shap_vizz(shap_dt_by_feature)
            st.plotly_chart(fig_dt, use_container_width=True)
        with graph_col_nn:
            st.header("🧠 Perpetrator Results")
            delta_color_nn,value_nn,delta_nn = metric_parser(perp)
            coldt1, coldt2, coldt3 = st.columns(3)
            coldt2.metric(label="Perpetrator Risk", value=value_nn, delta=delta_nn, delta_color=delta_color_nn)
            fig_nn = shap_vizz(shap_nn_by_feature)
            st.plotly_chart(fig_nn, use_container_width=True)
            @st.dialog("🌳 + 🧠 Overlap Results", width="large")
            def show_overlap():
                # your overlap parser and plotting logic
                delta_color_ov, value_ov, delta_ov = overlap_metric_parser(vict, perp)
                coldtnn1, coldtnn2, coldtnn3, coldtnn4 = st.columns([1,2,1,1])
                with coldtnn2:
                    st.metric(label="Overlap Risk",value=value_ov,delta=delta_ov,delta_color=delta_color_ov)
                with coldtnn3:
                    if value_dt == "High":
                        st.metric(label="Victim Risk",value="⚠️", delta=delta_dt, delta_color=delta_color_dt)
                    else:
                        st.metric(label="Victim Risk",value="✅", delta=delta_dt, delta_color=delta_color_dt)
                with coldtnn4:
                    if value_nn == "High":
                        st.metric(label="Perpetrator Risk", value="⚠️", delta=delta_nn, delta_color=delta_color_nn)
                    else:
                        st.metric(label="Perpetrator Risk", value="✅", delta=delta_nn, delta_color=delta_color_nn)
                fig_ov = shap_vizz(overlap_shap)
                st.plotly_chart(fig_ov, use_container_width=True)

            # 3) Trigger it from a button in your two-column layout
            cccal1,cccal2 = st.columns([1,2])
            with cccal2:
                overlap_button = st.button(
                    "🧬 Check Overlap Features 🌳 + 🧠",
                    use_container_width=True,
                    type="primary"
                )
                if overlap_button:
                    show_overlap()
    with tab_admin:            
        # 6) Display in main area
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            st.subheader("🌐 Model Variables")
            st.json(model_vars_obj)

        with col2:
            st.subheader("⚙️ PCA Projection")
            st.json(pca_obj)

        with col3:
            st.subheader("🧠 Predictions + SHAP")
            st.json(prediction_results)

else:
    col1,col2 = st.columns([2,3])
    with col1:
        tab00, tab0, tab1, tab2, tab3, tab4 = st.tabs(["🌍 Demographic","🌱 Habits","📒 Self-Efficacy", "☕ Social-Support", "✒️ Moral", "🌡️ Impulsivity"])
        with tab00:
            ccol1,ccol2 = st.columns(2)
            with ccol1:
                country = st.selectbox(
                    label = "COUNTRY",
                    options = ["--"] + EUROPEAN_COUNTRIES + AFRICAN_COUNTRIES + AMERICAN_COUNTRIES + ASIAN_COUNTRIES + OCEANIA_COUNTRIES,
                )
                ethnic = st.selectbox(
                    label = "ETHNIC",
                    options = ["--"] + ETHNIC_GROUPS["EUROPEAN"] + ETHNIC_GROUPS["AFRICAN"] + ETHNIC_GROUPS["ASIAN"] + ETHNIC_GROUPS["AMERICAN"] + ETHNIC_GROUPS["MULTI_REGIONAL"] + ETHNIC_GROUPS["OCEANIAN"]
                )
                sex = st.selectbox(
                    label = "BIOLOGICAL SEX",
                    options = ["--"] + ["MALE","FAMALE"],
                )
                orientation = st.selectbox(
                    label = "SEXUAL ORIENTATION",
                    options = ["--"] + SEXUAL_ORIENTATIONS,
                )
                age = st.slider(
                    "How old are you?",
                    min_value=13,
                    max_value=18,
                    value=13,    
                    step=1
                )
            with ccol2:
                pass
        with tab0:
            family = st.multiselect(
                "Indicate all individuals you regularly live with",
                ["Mother", "Father", "Mother's Partner", "Father's Partner", "Siblings/Stepsiblings", "Aunt/Oncle", "Grandparents", "Center"],
            )            
            escape = st.radio(
                "Have you committed an escape (or jailbreak) in the past year?",
                ["Yes", "No"],
                index=None, 
            )
            abuso_subs1 = st.radio(
                "In the past year, how often have you consumed alcoholic beverages?",
                ["Never", "A few times a year", "Monthly", "Weekly", "Daily or almost daily"],
                index=None,
                key="alcohol_frequency"
            )
            abuso_subs2 = st.radio(
                "In the past year, how often have you consumed 5 or more alcoholic drinks (beer, wine, vodka, cocktails, shots, etc.) in a single day?",
                ["Never", "A few times a year", "Monthly", "Weekly", "Daily or almost daily"],
                index=None,
                key="binge_drinking_frequency"
            )
            porno1 = st.radio(
                "In the past year, how often have you watched pornographic movies?",
                [
                    "Never",
                    "A few times a year",  # "Alguna vez al año" more naturally translates to this
                    "Monthly",
                    "Weekly",
                    "Daily or almost daily"
                ],
                index=None,  # No default selection
                key="porn_movies_frequency"
            )
            porno2 = st.radio(
                "In the past year, how often have you visited pornographic websites?",
                [
                    "Never",
                    "A few times a year",
                    "Monthly",
                    "Weekly",
                    "Daily or almost daily"
                ],
                index=None,
                key="porn_sites_frequency"
            )        
        with tab1:
            autoefic1 = st.radio(
                "When facing difficulties, I find ways to overcome them.",
                [
                    "False",
                    "More false than true",
                    "More true than false", 
                    "True"
                ],
                index=None,
                key="self_efficacy_1"
            )
            autoefic2 = st.radio(
                "I handle myself well even when unexpected events occur.",
                [
                    "False",
                    "More false than true",
                    "More true than false",
                    "True"
                ],
                index=None,
                key="self_efficacy_2"
            )
            autoefic3 = st.radio(
                "I always know what to do in unexpected circumstances.",
                [
                    "False", 
                    "More false than true",
                    "More true than false",
                    "True"
                ],
                index=None,
                key="self_efficacy_3"
            )
            autoefic4 = st.radio(
                "I'm good at achieving my goals and purposes.",
                [
                    "False",
                    "More false than true",
                    "More true than false",
                    "True"
                ],
                index=None,
                key="self_efficacy_4"
            )
            autoefic5 = st.radio(
                "No matter what happens, I'll be okay.",
                [
                    "False",
                    "More false than true",
                    "More true than false",
                    "True"
                ],
                index=None,
                key="self_efficacy_5"
            )
        with tab2:
            apoyo1 = st.radio(
                "There are adults I can talk to about my problems.",
                ["False", "More false than true", "More true than false", "True"],
                index=None,
                key="support_1"
            )
            apoyo2 = st.radio(
                "I have one or more good friends.",
                ["False", "More false than true", "More true than false", "True"],
                index=None,
                key="support_2"
            )
            apoyo3 = st.radio(
                "Among the adults I know, there are some I admire.",
                ["False", "More false than true", "More true than false", "True"],
                index=None,
                key="support_3"
            )
            apoyo4 = st.radio(
                "I have friends I can trust.",
                ["False", "More false than true", "More true than false", "True"],
                index=None,
                key="support_4"
            )
            apoyo5 = st.radio(
                "I talk about my problems with adults.",
                ["False", "More false than true", "More true than false", "True"],
                index=None,
                key="support_5"
            )
            apoyo6 = st.radio(
                "I get along well with my friends.",
                ["False", "More false than true", "More true than false", "True"],
                index=None,
                key="support_6"
            )
            apoyo7 = st.radio(
                "There are adults I can trust.",
                ["False", "More false than true", "More true than false", "True"],
                index=None,
                key="support_7"
            )
        with tab3:
            moral1 = st.radio(
                "I feel guilty if I've hurt a classmate.",
                ["Strongly disagree", "Somewhat disagree", "Neutral", "Somewhat agree", "Strongly agree"],
                index=None,
                key="moral_guilt"
            )
            moral2 = st.radio(
                "I regret it when I do something that hurts someone else.",
                ["Strongly disagree", "Somewhat disagree", "Neutral", "Somewhat agree", "Strongly agree"],
                index=None,
                key="moral_regret"
            )
            moral3 = st.radio(
                "I feel ashamed if people notice I've done something bad to someone.",
                ["Strongly disagree", "Somewhat disagree", "Neutral", "Somewhat agree", "Strongly agree"],
                index=None,
                key="moral_shame"
            )
            moral4 = st.radio(
                "I feel proud when I do something good for someone.",
                ["Strongly disagree", "Somewhat disagree", "Neutral", "Somewhat agree", "Strongly agree"],
                index=None,
                key="moral_pride"
            )
            moral5 = st.radio(
                "I feel bad when someone tells me I've hurt them.",
                ["Strongly disagree", "Somewhat disagree", "Neutral", "Somewhat agree", "Strongly agree"],
                index=None,
                key="moral_remorse"
            )
        with tab4:
            impuls1 = st.radio(
                "I plan what I need to do.",
                ["Rarely or never", "Occasionally", "Often", "Always or almost always"],
                index=None,
                key="planning"
            )
            impuls2 = st.radio(
                "I do things without thinking.",
                ["Rarely or never", "Occasionally", "Often", "Always or almost always"],
                index=None,
                key="acting_without_thinking"
            )
            impuls3 = st.radio(
                "I don't pay attention to things.",
                ["Rarely or never", "Occasionally", "Often", "Always or almost always"],
                index=None,
                key="inattention"
            )
            impuls4 = st.radio(
                "I'm a person with good self-control.",
                ["Rarely or never", "Occasionally", "Often", "Always or almost always"],
                index=None,
                key="self_control"
            )
            impuls5 = st.radio(
                "I concentrate easily.",
                ["Rarely or never", "Occasionally", "Often", "Always or almost always"],
                index=None,
                key="concentration"
            )
            impuls6 = st.radio(
                "I like to think things through carefully.",
                ["Rarely or never", "Occasionally", "Often", "Always or almost always"],
                index=None,
                key="deliberation"
            )
            impuls7 = st.radio(
                "I say things without thinking.",
                ["Rarely or never", "Occasionally", "Often", "Always or almost always"],
                index=None,
                key="verbal_impulsivity"
            )
            impuls8 = st.radio(
                "I act on the spur of the moment.",
                ["Rarely or never", "Occasionally", "Often", "Always or almost always"],
                index=None,
                key="spontaneity"
            )
    with col2:
        if impuls8:
            feat_pc_importance_df = load_csv_data(feature_pc_weights_path).drop(columns="Unnamed: 0")
            quiz = update_dict_vars(
                dict_vars,
                country,
                EUROPEAN_COUNTRIES,
                ethnic,
                sex,
                orientation,
                age,
                family,
                escape,
                abuso_subs1,
                abuso_subs2,
                porno1,
                porno2,
                autoefic1,
                autoefic2,
                autoefic3,
                autoefic4,
                autoefic5,
                impuls1,
                impuls2,
                impuls3,
                impuls4,
                impuls5,
                impuls6,
                impuls7,
                impuls8,
                apoyo1,
                apoyo2,
                apoyo3,
                apoyo4,
                apoyo5,
                apoyo6,
                apoyo7,
                moral1,
                moral2,
                moral3,
                moral4,
                moral5
            )
                # 2) Transform to PCA (returns JSON string) and parse
            pca_json_str = transform_json_to_pca(quiz)
            pca_list = json.loads(pca_json_str)

            # 3) Run classifiers (each returns JSON string) and parse
            vict_json_str = classify_dt_pcs(pca_list)
            perp_json_str = classify_pcs_nn(pca_list)
            vict = json.loads(vict_json_str)[0] if vict_json_str else {}
            perp = json.loads(perp_json_str)[0] if perp_json_str else {}

            # 4) Slice PC dicts for SHAP
            pcs_full = pca_list[0]
            pc_dict_dt = {f"PC{i}": pcs_full[f"PC{i}"] for i in range(1, 19)}
            pc_dict_nn = {f"PC{i}": pcs_full[f"PC{i}"] for i in range(1, 23)}

            # 5) Compute SHAP values
            shap_dt = infer_pc_shap_dt(pc_dict_dt)
            shap_nn = infer_pc_shap_nn(pc_dict_nn)

            shap_dt_by_feature = shap_original_from_pcs(feat_pc_importance_df, shap_dt)
            shap_nn_by_feature = shap_original_from_pcs(feat_pc_importance_df, shap_nn)
            
            shap_dt_by_feature = shap_to_percent_with_sign(shap_dt_by_feature)
            shap_nn_by_feature = shap_to_percent_with_sign(shap_nn_by_feature)
            # --- Combined results JSON (what you want to show + download) ---
            
            comb_shap = combine_shap_percent(shap_dt_by_feature,shap_nn_by_feature)
            comb_shap_filtered = { key: val_sign[0] for key, val_sign in comb_shap.items() }
            comb_shap_scaled = shap_to_percent_with_sign(comb_shap_filtered)
            overlap_shap = rebuild_shap_dict(comb_shap_scaled,comb_shap)
            
            prediction_results = {
                "Decision Tree": vict,
                "Neural Network": perp,
                "SHAP_DT_pcs": shap_dt,
                "SHAP_NN_pcs": shap_nn,
                "SHAP_DT_vars": shap_dt_by_feature,
                "SHAP_NN_vars": shap_nn_by_feature,
                "SHAP_overlap": overlap_shap
            }
            # Sidebar: download buttons
            with st.sidebar:
                # Model vars: extract single dict and preserve accents
                model_vars_obj = quiz
                model_vars_str = json.dumps(model_vars_obj, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download Vars JSON",
                    data=model_vars_str,
                    file_name="model_vars.json",
                    mime="application/json",
                    use_container_width=True,
                    type = 'tertiary'
                )

                # PCA: extract single dict and pretty-print
                pca_obj = pca_list[0]
                pca_str = json.dumps(pca_obj, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download PCA JSON",
                    data=pca_str,
                    file_name="pca_data.json",
                    mime="application/json",
                    use_container_width=True,
                    type = 'tertiary'
                )

                # Combined predictions + SHAP
                st.download_button(
                    label="Download Results",
                    data=json.dumps(prediction_results, indent=2, ensure_ascii=False),
                    file_name="prediction_results.json",
                    mime="application/json",
                    use_container_width=True,
                    type = 'primary'
                )

            ccol1,ccol2 = st.columns(2)
            with ccol1:
                st.header("🌳 Victim Results")
                delta_color_dt,value_dt,delta_dt = metric_parser(vict)
                coldt1, coldt2, coldt3 = st.columns(3)
                coldt2.metric(label="Victim Risk", value=value_dt, delta=delta_dt, delta_color=delta_color_dt)
                fig_dt = shap_vizz(shap_dt_by_feature)
                st.plotly_chart(fig_dt, use_container_width=True)
            with ccol2:
                st.header("🧠 Perpetrator Results")
                delta_color_nn,value_nn,delta_nn = metric_parser(perp)
                coldt1, coldt2, coldt3 = st.columns(3)
                coldt2.metric(label="Perpetrator Risk", value=value_nn, delta=delta_nn, delta_color=delta_color_nn)
                fig_nn = shap_vizz(shap_nn_by_feature)
                st.plotly_chart(fig_nn, use_container_width=True)
                @st.dialog("🌳 + 🧠 Overlap Results", width="large")
                def show_overlap():
                    # your overlap parser and plotting logic
                    delta_color_ov, value_ov, delta_ov = overlap_metric_parser(vict, perp)
                    coldtnn1, coldtnn2, coldtnn3, coldtnn4 = st.columns([1,2,1,1])
                    with coldtnn2:
                        st.metric(label="Overlap Risk",value=value_ov,delta=delta_ov,delta_color=delta_color_ov)
                    with coldtnn3:
                        if value_dt == "High":
                            st.metric(label="Victim Risk",value="⚠️", delta=delta_dt, delta_color=delta_color_dt)
                        else:
                            st.metric(label="Victim Risk",value="✅", delta=delta_dt, delta_color=delta_color_dt)
                    with coldtnn4:
                        if value_nn == "High":
                            st.metric(label="Perpetrator Risk", value="⚠️", delta=delta_nn, delta_color=delta_color_nn)
                        else:
                            st.metric(label="Perpetrator Risk", value="✅", delta=delta_nn, delta_color=delta_color_nn)
                    fig_ov = shap_vizz(overlap_shap)
                    st.plotly_chart(fig_ov, use_container_width=True)

                # 3) Trigger it from a button in your two-column layout
                cccal1,cccal2 = st.columns([1,2])
                with cccal2:
                    overlap_button = st.button(
                        "🧬 Check Overlap Features 🌳 + 🧠",
                        use_container_width=True,
                        type="primary"
                    )
                    if overlap_button:
                        show_overlap()


