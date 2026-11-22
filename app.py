
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import sklearn


# ============================================================
# HELPER: HAPUS INDENTASI HTML
# Markdown menganggap baris berindentasi >= 4 spasi sebagai
# code block. Fungsi ini menghapus indentasi & baris kosong
# agar HTML dirender dengan benar.
# ============================================================

def html(text: str) -> str:
    return "\n".join(
        line.strip()
        for line in text.strip().splitlines()
        if line.strip()
    )


def md_html(text: str):
    st.markdown(html(text), unsafe_allow_html=True)


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

TRAINING_RANGES = {
    "age": (0.08, 82.0),
    "avg_glucose_level": (55.12, 271.74),
    "bmi": (10.3, 97.6),
}

TEST_METRICS = {
    "Accuracy": "84.64%",
    "Precision": "21.08%",
    "Recall": "78.00%",
    "F1-Score": "0.3319",
    "ROC-AUC": "0.8410",
    "PR-AUC": "0.2569",
}

CONFUSION_COUNTS = {
    "true_negative": 826,
    "false_positive": 146,
    "false_negative": 11,
    "true_positive": 39,
}


# ============================================================
# CUSTOM STYLING
# ============================================================

md_html(
    """
    <style>

    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');

    :root {
        color-scheme: light;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 85% 2%, rgba(30, 145, 130, 0.10), transparent 28%),
            linear-gradient(180deg, #F5FAF9 0%, #F7F9FC 55%, #F4F7FA 100%) !important;
        color: #183D46 !important;
    }

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="stMainBlockContainer"] {
        color: #183D46 !important;
    }

    [data-testid="stMain"] .stMarkdown p,
    [data-testid="stMain"] .stMarkdown li {
        color: #31545B;
    }

    [data-testid="stMain"] h1,
    [data-testid="stMain"] h2,
    [data-testid="stMain"] h3,
    [data-testid="stMain"] h4 {
        color: #183D46 !important;
    }

    [data-testid="stMain"] [data-testid="stWidgetLabel"],
    [data-testid="stMain"] [data-testid="stWidgetLabel"] *,
    [data-testid="stMain"] [data-testid="stWidgetLabel"] p,
    [data-testid="stMain"] .stSelectbox label,
    [data-testid="stMain"] .stNumberInput label {
        color: #183D46 !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }

    div[data-baseweb="select"] > div {
        background-color: #262832 !important;
        border: 1px solid #383B46 !important;
        border-radius: 10px !important;
    }

    div[data-baseweb="select"] > div * {
        color: #FFFFFF !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background-color: #FFFFFF !important;
    }

    [role="option"],
    [role="option"] * {
        color: #183D46 !important;
    }

    div[data-testid="stNumberInput"] input {
        background-color: #262832 !important;
        color: #FFFFFF !important;
        -webkit-text-fill-color: #FFFFFF !important;
        border-radius: 8px !important;
    }

    div[data-testid="stNumberInput"] button {
        color: #FFFFFF !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #123B43 0%, #102F3B 100%) !important;
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
        color: #FFFFFF !important;
        margin-bottom: 1.25rem;
        box-shadow: 0 14px 35px rgba(18, 59, 67, 0.13);
    }

    .hero h1 {
        color: #FFFFFF !important;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(1.8rem, 3vw, 2.55rem);
        line-height: 1.15;
        margin: 0.35rem 0 0.55rem 0;
    }

    .hero p {
        color: #E5F4F1 !important;
        margin-bottom: 0;
        font-size: 0.98rem;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 0.13em;
        font-size: 0.72rem;
        font-weight: 700;
        color: #B8E5D9 !important;
    }

    .section-title {
        font-family: 'Manrope', sans-serif;
        color: #183D46 !important;
        font-weight: 800;
        font-size: 1.15rem;
        margin: 0.4rem 0 0.75rem 0;
    }

    .soft-card {
        background: #FFFFFF !important;
        border: 1px solid #DCE9E7;
        border-radius: 17px;
        padding: 1rem 1.15rem;
        box-shadow: 0 7px 22px rgba(20, 55, 60, 0.045);
        min-height: 112px;
        color: #183D46 !important;
    }

    .soft-card .label {
        color: #526B73 !important;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.45rem;
    }

    .soft-card .value {
        color: #173E46 !important;
        font-family: 'Manrope', sans-serif;
        font-size: 1.65rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .soft-card .hint {
        color: #627D84 !important;
        font-size: 0.75rem;
        margin-top: 0.35rem;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF !important;
        border: 1px solid #DCE9E7 !important;
        border-radius: 15px;
        padding: 0.8rem 1rem;
        box-shadow: 0 5px 18px rgba(20, 55, 60, 0.035);
        color: #183D46 !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] *,
    [data-testid="stMetricLabel"] p {
        color: #526B73 !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] *,
    [data-testid="stMetricValue"] div {
        color: #173E46 !important;
        opacity: 1 !important;
        font-weight: 800 !important;
    }

    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] * {
        color: #187A68 !important;
    }

    .result-card {
        background: #FFFFFF !important;
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
        color: #526B73 !important;
        font-size: 0.82rem;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    [data-testid="stAlert"] p,
    [data-testid="stAlert"] div,
    [data-testid="stAlert"] span {
        color: #183D46 !important;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background: linear-gradient(90deg, #17645F 0%, #238879 100%) !important;
        color: #FFFFFF !important;
        border: 0;
        border-radius: 11px;
        min-height: 2.8rem;
        font-weight: 700;
        width: 100%;
    }

    div.stButton > button:hover,
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: linear-gradient(90deg, #124E4B 0%, #176E63 100%) !important;
        color: #FFFFFF !important;
        border: 0;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #DCE9E7;
        border-radius: 10px;
    }

    .footer-note {
        color: #627D84 !important;
        font-size: 0.76rem;
        border-top: 1px solid #DDE8E6;
        padding-top: 0.8rem;
        margin-top: 1.8rem;
    }

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """
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
        "stroke_prediction_model.pkl di folder yang sama dengan app.py."
    )
    st.stop()


try:
    model_package = load_model_package(str(MODEL_PATH))

except Exception as exc:
    st.error(
        "Model gagal dimuat. Pastikan versi Scikit-learn sesuai "
        "dengan requirements.txt (scikit-learn==1.6.1)."
    )

    with st.expander("Detail error teknis"):
        st.code(str(exc))

    st.stop()


required_keys = {"model", "threshold", "model_name", "features"}

if not required_keys.issubset(model_package.keys()):
    st.error("Struktur file model tidak sesuai.")
    st.stop()


model = model_package["model"]
threshold = float(model_package["threshold"])
model_name = str(model_package["model_name"])
feature_order = list(model_package["features"])


if feature_order != EXPECTED_FEATURES:
    st.warning(
        "Urutan fitur berbeda dari catatan notebook. "
        "Aplikasi mengikuti urutan yang tersimpan dalam model."
    )


if not 0 <= threshold <= 1:
    st.error("Threshold berada di luar rentang 0–1.")
    st.stop()


if not hasattr(model, "predict_proba"):
    st.error("Model tidak mendukung predict_proba().")
    st.stop()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:

    md_html(
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
        """
    )

    page = st.radio(
        "NAVIGASI",
        ["Dashboard", "Prediksi", "Tentang Model"],
        label_visibility="visible",
    )

    st.markdown("---")

    st.caption("PROJECT")

    st.markdown("**Stroke Prediction**")

    md_html(
        """
        <span style="font-size:0.78rem;color:#C4D9D7;">
            Model: Logistic Regression<br>
            Dataset: Healthcare Stroke Dataset
        </span>
        """
    )

    st.markdown("---")

    st.caption("Aplikasi akademik. Bukan alat diagnosis medis.")


# ============================================================
# REUSABLE COMPONENTS
# ============================================================

def render_hero(eyebrow: str, title: str, description: str):
    md_html(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """
    )


def render_metric_card(label: str, value: str, hint: str):
    md_html(
        f"""
        <div class="soft-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="hint">{hint}</div>
        </div>
        """
    )


def section_title(text: str):
    md_html(f'<div class="section-title">{text}</div>')


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

    section_title("Ringkasan Proyek")

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

        section_title("Tentang Aplikasi")

        md_html(
            """
            <div class="soft-card">
                <p style="color:#31545B;margin-top:0;">
                    Aplikasi ini menerima karakteristik demografis
                    dan kondisi kesehatan sebagai input, kemudian
                    menjalankannya melalui pipeline preprocessing
                    dan model Logistic Regression.
                </p>
                <p style="color:#31545B;margin-bottom:0;">
                    Pipeline yang tersimpan mencakup imputasi median
                    untuk fitur numerik, imputasi modus untuk fitur
                    kategorikal, One-Hot Encoding, serta StandardScaler
                    pada fitur numerik.
                </p>
            </div>
            """
        )

    with right:

        section_title("Metrik pada Data Testing")

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
        "Interpretasi: pada data testing proyek, Recall kelas stroke "
        "sebesar 78% (39 dari 50 observasi stroke terdeteksi). "
        "Precision sebesar 21,08% menunjukkan masih terdapat "
        "false positive yang cukup banyak."
    )

    md_html(
        """
        <div class="footer-note">
            Angka evaluasi berasal dari pengujian model pada test set
            di notebook proyek. Hasil ini bukan validasi klinis.
        </div>
        """
    )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "Prediksi":

    render_hero(
        "PREDICTION WORKSPACE",
        "Masukkan Data Pasien",
        "Lengkapi seluruh variabel di bawah ini untuk menjalankan "
        "klasifikasi menggunakan model yang telah dilatih.",
    )

    section_title("Informasi Demografis")

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
                help="Rentang usia dataset: 0,08–82 tahun.",
            )

            ever_married = st.selectbox("Pernah menikah?", ["Yes", "No"])

            work_type = st.selectbox(
                "Jenis pekerjaan",
                ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
            )

            residence_type = st.selectbox("Tipe tempat tinggal", ["Urban", "Rural"])

        with col2:

            hypertension_label = st.selectbox("Memiliki hipertensi?", ["Tidak", "Ya"])

            heart_disease_label = st.selectbox("Memiliki penyakit jantung?", ["Tidak", "Ya"])

            glucose = st.number_input(
                "Average glucose level",
                min_value=1.0,
                max_value=600.0,
                value=91.9,
                step=0.1,
                help="Rentang dataset: 55,12–271,74.",
            )

            bmi = st.number_input(
                "BMI",
                min_value=1.0,
                max_value=150.0,
                value=28.1,
                step=0.1,
                help="Rentang dataset: 10,3–97,6.",
            )

            smoking_status = st.selectbox(
                "Status merokok",
                ["Unknown", "never smoked", "formerly smoked", "smokes"],
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

        input_df = pd.DataFrame([input_values], columns=feature_order)

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
                "Beberapa input berada di luar rentang dataset "
                "pelatihan: "
                + ", ".join(out_of_range)
                + ". Prediksi perlu ditafsirkan dengan hati-hati."
            )

        try:

            score = float(model.predict_proba(input_df)[0, 1])

            predicted_class = int(score >= threshold)

            st.markdown("---")

            section_title("Hasil Klasifikasi Model")

            if predicted_class == 1:

                md_html(
                    """
                    <div class="result-card result-positive">
                        <div class="small-muted">HASIL KLASIFIKASI</div>
                        <h2 style="color:#A94E35;margin:0.35rem 0;">Kelas Stroke (1)</h2>
                        <p style="color:#5E7075;margin-bottom:0;">
                            Berdasarkan threshold model, input ini
                            diklasifikasikan ke kelas 1.
                        </p>
                    </div>
                    """
                )

            else:

                md_html(
                    """
                    <div class="result-card result-negative">
                        <div class="small-muted">HASIL KLASIFIKASI</div>
                        <h2 style="color:#187A68;margin:0.35rem 0;">Kelas Tidak Stroke (0)</h2>
                        <p style="color:#5E7075;margin-bottom:0;">
                            Berdasarkan threshold model, input ini
                            diklasifikasikan ke kelas 0.
                        </p>
                    </div>
                    """
                )

            score_col, threshold_col = st.columns(2)

            with score_col:
                st.metric("Skor keluaran model", f"{score:.4f}")

            with threshold_col:
                st.metric("Threshold klasifikasi", f"{threshold:.4f}")

            st.progress(min(max(score, 0.0), 1.0))

            st.caption(
                "Skor di atas adalah keluaran predict_proba dari model. "
                "Skor ini bukan probabilitas risiko klinis yang "
                "tervalidasi dan bukan diagnosis."
            )

            st.info(
                "Penting: hasil ini hanya demonstrasi model Machine "
                "Learning berdasarkan pola dataset Kaggle. Hasil "
                "tidak boleh digunakan untuk mendiagnosis, "
                "menyingkirkan, atau menentukan penanganan stroke. "
                "Untuk keluhan kesehatan, konsultasikan dengan "
                "tenaga medis."
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
        "Informasi algoritma, fitur, threshold, dan hasil evaluasi "
        "yang dilaporkan pada notebook proyek.",
    )

    section_title("Konfigurasi Model")

    info1, info2, info3 = st.columns(3)

    with info1:
        st.metric("Algoritma", model_name)

    with info2:
        st.metric("Threshold", f"{threshold:.4f}")

    with info3:
        st.metric("Scikit-learn", sklearn.__version__)

    section_title("Variabel yang Digunakan")

    st.write(
        "Gender, age, hypertension, heart_disease, ever_married, "
        "work_type, Residence_type, avg_glucose_level, bmi, "
        "dan smoking_status."
    )

    section_title("Evaluasi pada Test Set")

    metrics_df = pd.DataFrame(
        [{"Metrik": key, "Nilai": value} for key, value in TEST_METRICS.items()]
    )

    st.dataframe(metrics_df, width="stretch", hide_index=True)

    section_title("Confusion Matrix")

    cm_df = pd.DataFrame(
        [
            [CONFUSION_COUNTS["true_negative"], CONFUSION_COUNTS["false_positive"]],
            [CONFUSION_COUNTS["false_negative"], CONFUSION_COUNTS["true_positive"]],
        ],
        index=["Aktual: Tidak Stroke", "Aktual: Stroke"],
        columns=["Prediksi: Tidak Stroke", "Prediksi: Stroke"],
    )

    st.dataframe(cm_df, width="stretch")

    # Tanpa indentasi agar tidak dianggap code block
    st.markdown(
        "\n".join(
            [
                "**Catatan interpretasi**",
                "",
                "- Recall 78% berarti 39 dari 50 observasi berlabel stroke "
                "pada test set berhasil diklasifikasikan sebagai kelas stroke.",
                "- Precision 21,08% menunjukkan bahwa sebagian besar "
                "prediksi kelas stroke pada test set merupakan false positive.",
                "- Threshold dipilih dari data validasi dengan memaksimalkan "
                "F1-Score, bukan dari data testing.",
                "- Hasil evaluasi berasal dari satu pembagian train-test "
                "pada dataset yang digunakan dalam proyek. Performa "
                "pada populasi atau data lain dapat berbeda.",
            ]
        )
    )

    st.warning(
        "Model ini merupakan proyek akademik dan belum melalui "
        "validasi eksternal maupun validasi klinis. Jangan gunakan "
        "sebagai pengganti penilaian tenaga kesehatan."
    )


# ============================================================
# FOOTER
# ============================================================

md_html(
    """
    <div class="footer-note">
        Stroke Prediction • Academic Machine Learning Project<br>
        Built with Streamlit and Scikit-learn.
    </div>
    """
)
