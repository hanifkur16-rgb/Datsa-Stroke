import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Load trained model
MODEL_PATH = Path(__file__).parent / "stroke_prediction_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model_package = load_model()

model = model_package["model"]
threshold = model_package["threshold"]
features = model_package["features"]

# Page configuration
st.set_page_config(
    page_title="Stroke Prediction",
    page_icon="🩺",
    layout="wide"
)


# ==========================================
# CUSTOM DESIGN - FIXED FONT CONTRAST
# ==========================================

st.markdown("""
<style>

/* ======================================
   MAIN BACKGROUND
====================================== */

:root {
    color-scheme: light;
}

.stApp,
[data-testid="stAppViewContainer"] {
    background-color: #F4F8F8 !important;
    color: #183D46 !important;
}

[data-testid="stAppViewContainer"] .main {
    color: #183D46 !important;
}


/* ======================================
   MAIN TEXT
====================================== */

[data-testid="stAppViewContainer"] .main p,
[data-testid="stAppViewContainer"] .main li,
[data-testid="stAppViewContainer"] .main span,
[data-testid="stAppViewContainer"] .main label,
[data-testid="stAppViewContainer"] .main
[data-testid="stMarkdownContainer"] {
    color: #294D55 !important;
    opacity: 1 !important;
}


/* ======================================
   HEADINGS
====================================== */

.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4,
.stApp h5,
.stApp h6 {
    color: #164E50 !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}


/* ======================================
   WIDGET LABELS
   Gender, Age, BMI, etc.
====================================== */

[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
[data-testid="stWidgetLabel"] p,
.stSelectbox label,
.stNumberInput label,
.stSelectbox label p,
.stNumberInput label p {
    color: #183D46 !important;
    opacity: 1 !important;
    font-weight: 600 !important;
}


/* ======================================
   SELECTBOX
====================================== */

[data-baseweb="select"] > div {
    background-color: #FFFFFF !important;
    border: 1px solid #C6D9D7 !important;
    border-radius: 10px !important;
}

[data-baseweb="select"] * {
    color: #183D46 !important;
}

[role="listbox"],
[role="option"] {
    background-color: #FFFFFF !important;
    color: #183D46 !important;
}


/* ======================================
   NUMBER INPUT
====================================== */

[data-testid="stNumberInput"] input {
    background-color: #FFFFFF !important;
    color: #183D46 !important;
    border: 1px solid #C6D9D7 !important;
    -webkit-text-fill-color: #183D46 !important;
}

[data-testid="stNumberInput"] button {
    color: #183D46 !important;
    background-color: #F0F6F5 !important;
}


/* ======================================
   METRIC CARDS
   Accuracy, Precision, Recall, etc.
====================================== */

[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DCE9E7 !important;
    border-radius: 14px !important;
    padding: 15px !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricLabel"] p {
    color: #607B80 !important;
    opacity: 1 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] *,
[data-testid="stMetricValue"] div {
    color: #164E50 !important;
    opacity: 1 !important;
    font-weight: 700 !important;
}

[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * {
    color: #187C75 !important;
}


/* ======================================
   DATAFRAME / TABLE
====================================== */

[data-testid="stDataFrame"] {
    color: #183D46 !important;
}

[data-testid="stDataFrame"] * {
    color: #183D46 !important;
}


/* ======================================
   ALERT BOX
====================================== */

[data-testid="stAlert"] {
    border-radius: 10px !important;
}

[data-testid="stAlert"] p,
[data-testid="stAlert"] span {
    color: #234D57 !important;
    opacity: 1 !important;
}


/* ======================================
   BUTTON
====================================== */

div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background-color: #187C75 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    height: 45px !important;
    font-weight: 700 !important;
}

div.stButton > button *,
div[data-testid="stFormSubmitButton"] > button * {
    color: #FFFFFF !important;
}

div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background-color: #12665F !important;
    color: #FFFFFF !important;
}


/* ======================================
   SIDEBAR
   Keep dark green design
====================================== */

[data-testid="stSidebar"] {
    background-color: #123B43 !important;
}

[data-testid="stSidebar"] * {
    color: #F2FAF8 !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    background-color: rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 8px;
}


/* ======================================
   CAPTION
====================================== */

[data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] * {
    color: #607B80 !important;
}


/* ======================================
   GENERAL IMPROVEMENTS
====================================== */

input,
textarea {
    color: #183D46 !important;
}

hr {
    border-color: #DCE9E7 !important;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🩺 Stroke Prediction")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Stroke Prediction", "About Model"]
)

# ==========================================
# DASHBOARD
# ==========================================

if page == "Dashboard":

    st.title("Stroke Prediction Dashboard")

    st.write(
        "Machine Learning Based Stroke Classification System"
    )

    st.info(
        "Aplikasi akademik untuk mengklasifikasikan "
        "status stroke berdasarkan karakteristik pasien."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Dataset", "5,110")
    col2.metric("Total Features", "10")
    col3.metric("Final Model", "Logistic Regression")

    st.subheader("Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", "84.64%")
    c2.metric("Precision", "21.08%")
    c3.metric("Recall", "78.00%")

    c4, c5, c6 = st.columns(3)

    c4.metric("F1-Score", "0.3319")
    c5.metric("ROC-AUC", "0.8410")
    c6.metric("PR-AUC", "0.2569")


# ==========================================
# STROKE PREDICTION
# ==========================================

elif page == "Stroke Prediction":

    st.title("Stroke Prediction")
    st.write("Enter patient information below.")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Female", "Male", "Other"]
            )

            age = st.number_input(
                "Age",
                min_value=0.08,
                max_value=120.0,
                value=45.0
            )

            hypertension = st.selectbox(
                "Hypertension",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            heart_disease = st.selectbox(
                "Heart Disease",
                [0, 1],
                format_func=lambda x: "Yes" if x == 1 else "No"
            )

            ever_married = st.selectbox(
                "Ever Married",
                ["Yes", "No"]
            )

        with col2:

            work_type = st.selectbox(
                "Work Type",
                [
                    "Private",
                    "Self-employed",
                    "Govt_job",
                    "children",
                    "Never_worked"
                ]
            )

            residence_type = st.selectbox(
                "Residence Type",
                ["Urban", "Rural"]
            )

            glucose = st.number_input(
                "Average Glucose Level",
                min_value=1.0,
                max_value=600.0,
                value=91.9
            )

            bmi = st.number_input(
                "BMI",
                min_value=1.0,
                max_value=150.0,
                value=28.1
            )

            smoking_status = st.selectbox(
                "Smoking Status",
                [
                    "Unknown",
                    "never smoked",
                    "formerly smoked",
                    "smokes"
                ]
            )

        submitted = st.form_submit_button(
            "PREDICT STROKE",
            use_container_width=True
        )

    if submitted:

        input_data = {
            "gender": gender,
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence_type,
            "avg_glucose_level": glucose,
            "bmi": bmi,
            "smoking_status": smoking_status
        }

        input_df = pd.DataFrame(
            [input_data],
            columns=features
        )

        try:

            prediction_score = model.predict_proba(
                input_df
            )[0][1]

            prediction = int(
                prediction_score >= threshold
            )

            st.subheader("Prediction Result")

            st.metric(
                "Model Output Score",
                f"{prediction_score:.4f}"
            )

            st.metric(
                "Classification Threshold",
                f"{threshold:.4f}"
            )

            if prediction == 1:

                st.error(
                    "Prediction: Stroke Class (1)"
                )

            else:

                st.success(
                    "Prediction: Non-Stroke Class (0)"
                )

            st.warning(
                "This application is an academic machine "
                "learning project, not a medical diagnosis "
                "or clinically validated risk assessment."
            )

        except Exception as e:

            st.error("Prediction error occurred.")
            st.exception(e)


# ==========================================
# ABOUT MODEL
# ==========================================

elif page == "About Model":

    st.title("About the Model")

    st.write(
        "This application uses Logistic Regression "
        "trained on the Stroke Prediction dataset."
    )

    st.subheader("Model Configuration")

    st.write("Algorithm: Logistic Regression")
    st.write("Solver: liblinear")
    st.write("C: 0.01")
    st.write(f"Threshold: {threshold:.4f}")

    st.subheader("Features")

    st.write(features)

    st.subheader("Evaluation Results")

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC",
            "PR-AUC"
        ],
        "Score": [
            0.8464,
            0.2108,
            0.7800,
            0.3319,
            0.8410,
            0.2569
        ]
    })

    st.dataframe(
        metrics,
        use_container_width=True,
        hide_index=True
    )

    st.warning(
        "The model has not undergone external "
        "clinical validation. Do not use it as "
        "a substitute for medical assessment."
    )

