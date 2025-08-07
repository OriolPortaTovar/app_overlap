
# 🔍 Violence Risk Screening App

This web application supports **victimology experts** and **survey respondents** by enabling risk screening for youth violence, through data analysis and predictive modeling.

---

## 📌 Key Objectives

- Help experts identify potential **victims**, **perpetrators**, or **both** based on behavioral survey data.
- Provide a transparent, data-driven methodology for **early detection and triage** of interpersonal violence.
- Offer **exploratory tools**, **predictive quizzes**, and **downloadable reports** to enhance usability and impact.

---

## 🧭 Application Structure

```
app/
├── .streamlit/         # Streamlit config
│   └── pages.toml
├── data/               # Feature importance and model data
├── data_samples/       # Sample survey files
├── docu/               # Documentation and schema
├── img/                # Logo and visuals
├── models/             # Predictive models (DT, NN, PCA)
├── pages/              # Main app tabs (EDA, prediction)
├── utils/              # Data processing and visualization
└── app.py              # Main entry point
```

## INSTALATION

cd D:\ICREA
python -m venv app_env
.\app_env\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

### Execute

streamlit run app.py

---

## 📊 EDA Tab: Exploratory Data Analysis

Allows users to:

- Upload training datasets
- Visualize raw and processed data
- Filter, inspect distributions, and detect missing values
- Perform **univariate analysis** using the target variables:
  - `VICTIM`
  - `PERPETRATOR`
  - `VICTIM_PERPETRATOR`
- Analyze relationships using a **frequentist approach**
- 📥 Download all outputs as **PNG** or **CSV**

---

## 🧠 Predictive Quiz Tab

Two prediction workflows:

1. **Upload a completed survey file** (e.g., Excel)
2. **Fill out the quiz manually in the app**

- Predictions are made using trained models (DT + NN)
- 📥 Full results downloadable, including **intermediate steps**

---

## 🧪 Predictive Modeling Summary

### Methodology

- Decomposed the tri-label problem (victim/perpetrator/both) into two **binary classifiers**:
  - 🌳 **Decision Tree (DT)** for detecting victims
  - 🧠 **Neural Network (NN)** for detecting perpetrators
- Ensemble model detects both roles when **both classifiers agree**
- High **recall** prioritized to **minimize missed cases**

### Performance Overview

| Metric         | Victim DT | Perpetrator NN | Overlap Ensemble |
|----------------|-----------|----------------|------------------|
| Accuracy       | 57.6%     | 47.1%          | 54.6%            |
| Precision      | 54.2%     | 29.5%          | 25.9%            |
| Recall         | 91.1%     | 90.0%          | 93.8%            |
| F1 Score       | 67.8%     | 44.5%          | 40.6%            |
| Specificity    | 24.9%     | 34.0%          | 46.9%            |
| Balanced Acc.  | 58.0%     | 62.0%          | 70.3%            |

---

## ⚙️ Tech Stack

- **Python 3.12**
- `pandas`, `numpy`, `scikit-learn`
- `TensorFlow`, `Keras`, `joblib`, `shap`
- **Streamlit** for the interactive web app

---

## 📚 Documentation

Located in the `/docu` folder:
- `ICREA_docu.pdf`
- `data_quiz_schema.xlsx`

---

## 🛡️ Purpose

This app is an **evidence-based prevention tool** designed to:

- Detect at-risk adolescents early
- Enable informed intervention by professionals
- Ensure transparency and reproducibility in violence-risk classification

---

**Developed for research and professional use in violence prevention.**
