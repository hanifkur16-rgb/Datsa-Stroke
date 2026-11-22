
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import sklearn


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Stroke Prediction | Healthcare AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "stroke_prediction_model.pkl"

# These are the exact feature names used in the uploaded notebook.
EXPECTED_FEATURES = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "smoking_status",
]

# Observed ranges in the source dataset. These are descriptive ranges,
# not medical reference ranges.
TRAINING_RANGES = {
    "age": (0.08, 82.0),
    "avg_glucose_level": (55.12, 271.74),
    "bmi": (10.3, 97.6),
}

# Metrics taken from the final Logistic Regression evaluation in the notebook.
TEST_METRICS = {
    "Accuracy": "84.64%",
    "Precision": "21.08%",
    "Recall": "78.00%",
    "F1-Score": "0.3319",
    "ROC-AUC": "0.8410",
    "PR-AUC": "0.2569",
}

# Confusion matrix reconstructed from the final test report:
# TN=826, FP=146, FN=11, TP=39.
CONFUSION_COUNTS = {
    "true_negative": 826,
    "false_positive": 146,
    "false_negative": 11,
    "true_positive": 39,
}


# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 2%, rgba(30, 145, 130, 0.10), transparent 28%),
            linear-gradient(180deg, #F5FAF9 0%, #F7F9FC 55%, #F4F7FA 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(245, 250, 249, 0.85);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #123B43 0%, #102F3B 100%);
    }

    [data-testid="stSidebar"] * {
        color: #F2FAF8;
    }

    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: rgba(255,255,255,0.055);
        padding: 0.55rem 0.75rem;
        border-radius: 10px;
        margin-bottom: 0.25rem;
    }

    .hero {
        background: linear-gradient(115deg, #123B43 0%, #17645F 58%, #25877B 100%);
        border-radius: 22px;
        padding: 1.8rem 2rem;
        color: white;
        margin-bottom: 1.25rem;
        box-shadow: 0 14px 35px rgba(18, 59, 67, 0.13);
    }

    .hero h1 {
        color: white;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.8rem, 3vw, 2.55rem);
        line-height: 1.15;
        margin: 0.35rem 0 0.55rem 0;
    }

    .hero p {
        color: rgba(255,255,255,0.84);
        margin-bottom: 0;
        font-size: 0.98rem;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.13em;
        font-size: 0.72rem;
        font-weight: 700;
        color: #B8E5D9;
    }

    .section-title {
        font-family: 'Manrope', sans-serif;
        color: #183D46;
        font-weight: 800;
        font-size: 1.15rem;
        margin: 0.4rem 0 0.75rem 0;
    }

    .soft-card {
        background: rgba(255,255,255,0.92);
        border: 1px solid #E2ECEA;
        border-radius: 17px;
        padding: 1rem 1.15rem;
        box-shadow: 0 7px 22px rgba(20, 55, 60, 0.045);
        min-height: 112px;
    }

    .soft-card .label {
        color: #688087;
        font-size: 0.78rem;
        font-weight: 600;
        margin-bottom: 0.45rem;
    }

    .soft-card .value {
        color: #173E46;
        font-family: 'Manrope', sans-serif;
        font-size: 1.65rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .soft-card .hint {
        color: #7A9095;
        font-size: 0.75rem;
        margin-top: 0.35rem;
    }

    .result-card {
        background: white;
        border: 1px solid #DFEAE8;
        border-radius: 18px;
        padding: 1.25rem 1.35rem;
        box-shadow: 0 8px 25px rgba(20, 55, 60, 0.06);
    }

    .result-positive {
        border-left: 6px solid #D97757;
    }

    .result-negative {
        border-left: 6px solid #21977C;
    }

    .small-muted {
        color: #6B8187;
        font-size: 0.82rem;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #17645F 0%, #238879 100%);
        color: white;
        border: 0;
        border-radius: 11px;
        min-height: 2.8rem;
        font-weight: 700;
        width: 100%;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(90deg, #124E4B 0%, #176E63 100%);
        color: white;
        border: 0;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #E2ECEA;
        border-radius: 15px;
        padding: 0.8rem 1rem;
        box-shadow: 0 5px 18px rgba(20, 55, 60, 0.035);
    }

    .footer-note {
        color: #71868B;
        font-size: 0.76rem;
        border-top: 1px solid #DDE8E6;
        padding-top: 0.8rem;
        margin-top: 1.8rem;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource(show_spinner="Memuat model prediksi...")
def load_model_package(path: str):
    return joblib.load(path)


if not MODEL_PATH.exists():
    st.error(
        "File model tidak ditemukan. Letakkan "
        "`stroke_prediction_model.pkl` di folder yang sama dengan `app.py`."
    )
    st.stop()

try:
    model_package = load_model_package(str(MODEL_PATH))
except Exception as exc:
    st.error(
        "Model gagal dimuat. Pastikan versi Scikit-learn sesuai dengan "
        "`requirements.txt` (scikit-learn==1.6.1), lalu deploy ulang."
    )
    with st.expander("Detail error teknis"):
        st.code(str(exc))
    st.stop()

required_keys = {"model", "threshold", "model_name", "features"}
if not required_keys.issubset(model_package.keys()):
    st.error(
        "Struktur file model tidak sesuai. File harus berisi "
        "`model`, `threshold`, `model_name`, dan `features`."
    )
    st.stop()

model = model_package["model"]
threshold = float(model_package["threshold"])
model_name = str(model_package["model_name"])
feature_order = list(model_package["features"])

if feature_order != EXPECTED_FEATURES:
    st.warning(
        "Urutan fitur pada file model berbeda dari urutan yang tercatat "
        "di notebook. Aplikasi akan mengikuti urutan fitur yang tersimpan "
        "di dalam file model."
    )

if not 0 <= threshold <= 1:
    st.error("Threshold pada file model berada di luar rentang 0–1.")
    st.stop()

if not hasattr(model, "predict_proba"):
    st.error("Model yang tersimpan tidak mendukung predict_proba().")
    st.stop()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1.1rem 0;">
            <div style="font-size:2.1rem;">🩺</div>
            <div style="font-family:Manrope,sans-serif;font-size:1.25rem;font-weight:800;">
                Stroke<span style="color:#8ED7C4;">Predict</span>
            </div>
            <div style="font-size:0.76rem;color:#C4D9D7;">
                Healthcare Machine Learning
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "NAVIGASI",
        ["Dashboard", "Prediksi", "Tentang Model"],
        label_visibility="visible",
    )

    st.markdown("---")
    st.caption("PROJECT")
    st.markdown("**Stroke Prediction**")
    st.markdown(
        "<span style='font-size:0.78rem;color:#C4D9D7;'>"
        "Model: Logistic Regression<br>"
        "Dataset: Healthcare Stroke Dataset"
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    st.caption("Aplikasi akademik. Bukan alat diagnosis medis.")


# ============================================================
# REUSABLE HEADER
# ============================================================
def render_hero(eyebrow: str, title: str, description: str):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_card(label: str, value: str, hint: str):
    st.markdown(
        f"""
        <div class="soft-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DASHBOARD PAGE
# ============================================================
if page == "Dashboard":
    render_hero(
        "HEALTHCARE • MACHINE LEARNING",
        "Stroke Prediction Dashboard",
        "Aplikasi demonstrasi klasifikasi status stroke menggunakan "
        "model Logistic Regression yang telah dilatih pada dataset Kaggle.",
    )

    st.markdown('<div class="section-title">Ringkasan Proyek</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("TOTAL DATASET", "5,110", "Observasi pada dataset")
    with c2:
        render_metric_card("FITUR PREDIKTOR", "10", "Variabel input model")
    with c3:
        render_metric_card("MODEL FINAL", "Logistic", "Logistic Regression")
    with c4:
        render_metric_card("THRESHOLD", f"{threshold:.4f}", "Ambang klasifikasi")

    st.write("")
    left, right = st.columns([1.25, 1])

    with left:
        st.markdown(
            '<div class="section-title">Tentang Aplikasi</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="soft-card">
                <p style="color:#31545B;margin-top:0;">
                    Aplikasi ini menerima karakteristik demografis dan kondisi
                    kesehatan sebagai input, kemudian menjalankannya melalui
                    pipeline preprocessing dan model Logistic Regression.
                </p>
                <p style="color:#31545B;margin-bottom:0;">
                    Pipeline yang tersimpan mencakup imputasi median untuk fitur
                    numerik, imputasi modus untuk fitur kategorikal, One-Hot
                    Encoding, serta StandardScaler pada fitur numerik.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div class="section-title">Metrik pada Data Testing</div>',
            unsafe_allow_html=True,
        )
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Accuracy", TEST_METRICS["Accuracy"])
            st.metric("Recall", TEST_METRICS["Recall"])
            st.metric("ROC-AUC", TEST_METRICS["ROC-AUC"])
        with m2:
            st.metric("Precision", TEST_METRICS["Precision"])
            st.metric("F1-Score", TEST_METRICS["F1-Score"])
            st.metric("PR-AUC", TEST_METRICS["PR-AUC"])

    st.info(
        "Interpretasi: pada data testing proyek, Recall kelas stroke sebesar "
        "78% (39 dari 50 observasi stroke terdeteksi). Precision sebesar "
        "21,08% menunjukkan masih terdapat false positive yang cukup banyak."
    )

    st.markdown(
        '<div class="footer-note">Angka evaluasi berasal dari pengujian model '
        'pada test set di notebook proyek. Hasil ini bukan validasi klinis.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# PREDICTION PAGE
# ============================================================
elif page == "Prediksi":
    render_hero(
        "PREDICTION WORKSPACE",
        "Masukkan Data Pasien",
        "Lengkapi seluruh variabel di bawah ini untuk menjalankan klasifikasi "
        "menggunakan model yang telah dilatih.",
    )

    st.markdown(
        '<div class="section-title">Informasi Demografis</div>',
        unsafe_allow_html=True,
    )

    with st.form("stroke_prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox(
                "Jenis kelamin",
                ["Female", "Male", "Other"],
                index=0,
                help="Kategori sesuai dataset pelatihan.",
            )
            age = st.number_input(
                "Usia (tahun)",
                min_value=0.08,
                max_value=120.0,
                value=45.0,
                step=1.0,
                help="Rentang usia pada dataset pelatihan: 0,08–82 tahun.",
            )
            ever_married = st.selectbox(
                "Pernah menikah?",
                ["Yes", "No"],
            )
            work_type = st.selectbox(
                "Jenis pekerjaan",
                [
                    "Private",
                    "Self-employed",
                    "Govt_job",
                    "children",
                    "Never_worked",
                ],
            )
            residence_type = st.selectbox(
                "Tipe tempat tinggal",
                ["Urban", "Rural"],
            )

        with col2:
            hypertension_label = st.selectbox(
                "Memiliki hipertensi?",
                ["Tidak", "Ya"],
            )
            heart_disease_label = st.selectbox(
                "Memiliki penyakit jantung?",
                ["Tidak", "Ya"],
            )
            glucose = st.number_input(
                "Average glucose level",
                min_value=1.0,
                max_value=600.0,
                value=91.9,
                step=0.1,
                help="Rentang pada dataset pelatihan: 55,12–271,74.",
            )
            bmi = st.number_input(
                "BMI",
                min_value=1.0,
                max_value=150.0,
                value=28.1,
                step=0.1,
                help="Rentang pada dataset pelatihan: 10,3–97,6.",
            )
            smoking_status = st.selectbox(
                "Status merokok",
                [
                    "Unknown",
                    "never smoked",
                    "formerly smoked",
                    "smokes",
                ],
            )

        st.markdown("---")
        submitted = st.form_submit_button(
            "Jalankan Prediksi",
            width="stretch",
        )

    if submitted:
        input_values = {
            "gender": gender,
            "age": float(age),
            "hypertension": 1 if hypertension_label == "Ya" else 0,
            "heart_disease": 1 if heart_disease_label == "Ya" else 0,
            "ever_married": ever_married,
            "work_type": work_type,
            "Residence_type": residence_type,
            "avg_glucose_level": float(glucose),
            "bmi": float(bmi),
            "smoking_status": smoking_status,
        }

        input_df = pd.DataFrame(
            [input_values],
            columns=feature_order,
        )

        # Flag values outside the observed ranges in the training dataset.
        out_of_range = []
        for feature, value in [
            ("age", age),
            ("avg_glucose_level", glucose),
            ("bmi", bmi),
        ]:
            low, high = TRAINING_RANGES[feature]
            if value < low or value > high:
                out_of_range.append(
                    f"{feature} ({value:g}; rentang data {low:g}–{high:g})"
                )

        if out_of_range:
            st.warning(
                "Beberapa input berada di luar rentang yang ditemukan pada "
                "dataset pelatihan: " + ", ".join(out_of_range) +
                ". Model tetap menghasilkan keluaran, tetapi prediksi "
                "di luar rentang data perlu ditafsirkan dengan sangat hati-hati."
            )

        try:
            score = float(model.predict_proba(input_df)[0, 1])
            predicted_class = int(score >= threshold)

            st.markdown("---")
            st.markdown(
                '<div class="section-title">Hasil Klasifikasi Model</div>',
                unsafe_allow_html=True,
            )

            if predicted_class == 1:
                st.markdown(
                    f"""
                    <div class="result-card result-positive">
                        <div class="small-muted">HASIL KLASIFIKASI</div>
                        <h2 style="color:#A94E35;margin:0.35rem 0;">
                            Kelas Stroke (1)
                        </h2>
                        <p style="color:#5E7075;margin-bottom:0;">
                            Berdasarkan threshold model, input ini
                            diklasifikasikan ke kelas 1.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card result-negative">
                        <div class="small-muted">HASIL KLASIFIKASI</div>
                        <h2 style="color:#187A68;margin:0.35rem 0;">
                            Kelas Tidak Stroke (0)
                        </h2>
                        <p style="color:#5E7075;margin-bottom:0;">
                            Berdasarkan threshold model, input ini
                            diklasifikasikan ke kelas 0.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            score_col, threshold_col = st.columns(2)
            with score_col:
                st.metric("Skor keluaran model", f"{score:.4f}")
            with threshold_col:
                st.metric("Threshold klasifikasi", f"{threshold:.4f}")

            st.progress(min(max(score, 0.0), 1.0))
            st.caption(
                "Skor di atas adalah keluaran predict_proba dari model. "
                "Skor ini bukan probabilitas risiko klinis yang tervalidasi "
                "dan bukan diagnosis."
            )

            st.info(
                "Penting: hasil ini hanya demonstrasi model Machine Learning "
                "berdasarkan pola dataset Kaggle. Hasil tidak boleh digunakan "
                "untuk mendiagnosis, menyingkirkan, atau menentukan penanganan "
                "stroke. Untuk keluhan atau kekhawatiran kesehatan, konsultasikan "
                "dengan tenaga medis."
            )

        except Exception as exc:
            st.error("Prediksi gagal dijalankan.")
            with st.expander("Detail error teknis"):
                st.code(str(exc))


# ============================================================
# ABOUT MODEL PAGE
# ============================================================
elif page == "Tentang Model":
    render_hero(
        "MODEL INFORMATION",
        "Tentang Model Prediksi",
        "Informasi algoritma, fitur, threshold, dan hasil evaluasi yang "
        "dilaporkan pada notebook proyek.",
    )

    st.markdown(
        '<div class="section-title">Konfigurasi Model</div>',
        unsafe_allow_html=True,
    )

    info1, info2, info3 = st.columns(3)
    with info1:
        st.metric("Algoritma", model_name)
    with info2:
        st.metric("Threshold", f"{threshold:.4f}")
    with info3:
        st.metric("Scikit-learn", sklearn.__version__)

    st.markdown(
        '<div class="section-title">Variabel yang Digunakan</div>',
        unsafe_allow_html=True,
    )
    st.write(
        "Gender, age, hypertension, heart_disease, ever_married, work_type, "
        "Residence_type, avg_glucose_level, bmi, dan smoking_status."
    )

    st.markdown(
        '<div class="section-title">Evaluasi pada Test Set</div>',
        unsafe_allow_html=True,
    )
    metrics_df = pd.DataFrame(
        [
            {"Metrik": key, "Nilai": value}
            for key, value in TEST_METRICS.items()
        ]
    )
    st.dataframe(metrics_df, width="stretch", hide_index=True)

    st.markdown(
        '<div class="section-title">Confusion Matrix</div>',
        unsafe_allow_html=True,
    )
    cm_df = pd.DataFrame(
        [
            [CONFUSION_COUNTS["true_negative"], CONFUSION_COUNTS["false_positive"]],
            [CONFUSION_COUNTS["false_negative"], CONFUSION_COUNTS["true_positive"]],
        ],
        index=["Aktual: Tidak Stroke", "Aktual: Stroke"],
        columns=["Prediksi: Tidak Stroke", "Prediksi: Stroke"],
    )
    st.dataframe(cm_df, width="stretch")

    st.markdown(
        """
        **Catatan interpretasi**

        - Recall 78% berarti 39 dari 50 observasi berlabel stroke pada test set
          berhasil diklasifikasikan sebagai kelas stroke.
        - Precision 21,08% menunjukkan bahwa sebagian besar prediksi kelas
          stroke pada test set merupakan false positive.
        - Threshold 0,6488 dipilih dari data validasi dengan memaksimalkan
          F1-Score, bukan dari data testing.
        - Hasil evaluasi berasal dari satu pembagian train-test pada dataset
          yang digunakan dalam proyek. Performa pada populasi atau data lain
          dapat berbeda.
        """
    )

    st.warning(
        "Model ini merupakan proyek akademik dan belum melalui validasi "
        "eksternal maupun validasi klinis. Jangan gunakan sebagai pengganti "
        "penilaian tenaga kesehatan."
    )


# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer-note">
        Stroke Prediction • Academic Machine Learning Project<br>
        Built with Streamlit and Scikit-learn.
    </div>
    """,
    unsafe_allow_html=True,
)
