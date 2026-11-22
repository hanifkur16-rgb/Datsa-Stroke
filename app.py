
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

# Custom design
st.markdown("""
<style>
.stApp {
    background-color: #F4F8F8;
}
h1, h2, h3 {
    color: #164E50;
}
div.stButton > button {
    background-color: #187C75;
    color: white;
    border-radius: 10px;
    height: 45px;
    font-weight: bold;
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

