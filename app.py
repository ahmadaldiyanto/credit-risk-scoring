import streamlit as st
import joblib
import pandas as pd
import plotly.graph_objects as go

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Credit Risk Scoring",
    page_icon="💳",
    layout="centered"
)

# ================= CUSTOM CSS (PRECISION SEPARATION) =================
st.markdown("""
<style>
    /* Background Utama */
    .stApp {
        background-color: #f1f5f9;
    }

    /* 1. KOTAK INPUT (FORM) */
    [data-testid="stForm"] {
        background: white !important;
        padding: 40px !important;
        border-radius: 20px !important;
        border: none !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05) !important;
        margin-bottom: 20px !important;
    }

    /* 2. KOTAK HASIL (TERPISAH TOTAL) */
    /* Menggunakan selektor anak langsung yang lebih ketat untuk menghindari 'root overlap' */
    div[data-testid="stVerticalBlock"] > div:has(div > .result-card-trigger) {
        background: white !important;
        padding: 40px !important;
        border-radius: 20px !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05) !important;
        margin-top: 20px !important;
        margin-bottom: 40px !important;
    }

    /* Menghilangkan background/shadow pada elemen anak di dalam kartu hasil */
    div:has(> .result-card-trigger) ~ div, 
    div:has(> .result-card-trigger) {
        background: transparent !important;
        box-shadow: none !important;
        border: none !important;
    }

    /* Tipografi */
    .main-title {
        text-align: center;
        font-size: 32px;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 5px;
    }
    .sub-title {
        text-align: center;
        color: #64748b;
        margin-bottom: 30px;
    }

    /* Header Section (Biru) */
    .section-header {
        font-size: 19px;
        font-weight: 700;
        color: #2563eb;
        margin-top: 25px;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 2px solid #f8fafc;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .result-title-text {
        font-size: 26px;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        gap: 12px;
        line-height: 1.2;
    }

    /* Button */
    .stButton > button {
        width: 100%;
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 12px !important;
        height: 52px;
        font-weight: 600;
        border: none;
        margin-top: 10px;
    }

    /* Marker */
    .result-card-trigger {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# ================= LOAD ASSETS =================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load("model.pkl")
        features = joblib.load("features.pkl")
        return model, features
    except:
        return None, None

model, feature_columns = load_assets()

# ================= HEADER =================
st.markdown("<div class='main-title'>💳 Credit Risk Scoring System</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Prediksi Risiko Gagal Bayar menggunakan Tuned XGBoost</div>", unsafe_allow_html=True)

# ================= INPUT FORM =================
with st.form("credit_risk_form"):
    
    st.markdown("<div class='section-header'>📋 Informasi Peminjam</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        loan_amnt = st.number_input("Jumlah Pinjaman (loan_amnt)", 500, 50000, 8000)
        int_rate = st.number_input("Suku Bunga % (int_rate)", 5.0, 30.0, 8.50)
        total_acc = st.number_input("Total Akun Kredit (total_acc)", 1, 100, 25)
        inq_last_6mths = st.number_input("Cek Kredit 6 Bln (inq_last_6mths)", 0, 10, 0)
        
    with col2:
        annual_inc = st.number_input("Pendapatan Tahunan (annual_inc)", 1000, 500000, 90000)
        dti = st.number_input("Rasio Utang (dti)", 0.0, 40.0, 10.50)
        open_acc = st.number_input("Akun Aktif (open_acc)", 1, 50, 12)

    st.markdown("<div class='section-header'>🏠 Detail Kredit</div>", unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    with col3:
        term = st.selectbox("Tenor (term)", ["36 months", "60 months"])
        home = st.selectbox("Status Rumah (home)", ["RENT", "OWN", "MORTGAGE"])
        state = st.selectbox("Wilayah (state)", ["CA", "FL", "CO", "NY", "TX"])
        
    with col4:
        grade = st.selectbox("Grade Kredit (grade)", [1, 2, 3, 4, 5, 6, 7])
        purpose = st.selectbox("Tujuan Pinjaman (purpose)", ["credit_card", "debt_consolidation", "small_business"])

    submit_button = st.form_submit_button("🔎 Analisis Risiko")

# ================= HASIL ANALISIS (KOTAK TERPISAH) =================
if submit_button:
    if model is None:
        st.error("File model.pkl atau features.pkl tidak ditemukan!")
    else:
        # Preprocessing & Feature Engineering
        input_dict = {col: 0 for col in feature_columns}
        input_dict.update({
            "loan_amnt": loan_amnt, "annual_inc": annual_inc, "int_rate": int_rate,
            "dti": dti, "total_acc": total_acc, "open_acc": open_acc,
            "inq_last_6mths": inq_last_6mths, "grade": grade,
            "loan_income_ratio": loan_amnt / (annual_inc + 1),
            "credit_exposure_ratio": total_acc / (open_acc + 1),
            "debt_load_ratio": dti * (loan_amnt / (annual_inc + 1)),
            "risk_interaction": int_rate * dti
        })
        
        # One-Hot Encoding
        if term == "60 months" and "term_ 60 months" in input_dict: 
            input_dict["term_ 60 months"] = 1
        if f"addr_state_{state}" in input_dict: 
            input_dict[f"addr_state_{state}"] = 1
        if f"purpose_{purpose}" in input_dict: 
            input_dict[f"purpose_{purpose}"] = 1
        if f"home_ownership_{home}" in input_dict: 
            input_dict[f"home_ownership_{home}"] = 1

        input_df = pd.DataFrame([input_dict])[feature_columns]
        prob = model.predict_proba(input_df)[0][1]

        # Menggunakan container untuk memastikan struktur DOM terisolasi
        with st.container():
            # Marker trigger disatukan dengan judul
            st.markdown("<div class='result-title-text'><span class='result-card-trigger'></span>📊 Hasil Analisis Risiko</div>", unsafe_allow_html=True)
            
            # Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number", 
                value=prob * 100, 
                number={'suffix': "%", 'font': {'size': 44, 'color': '#1e293b'}},
                title={'text': "Probabilitas Gagal Bayar", 'font': {'size': 20, 'color': '#64748b'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#cbd5e1"},
                    'bar': {'color': "#dc2626" if prob > 0.5 else "#16a34a"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "#f1f5f9",
                    'steps': [
                        {'range': [0, 40], 'color': "#dcfce7"},
                        {'range': [40, 70], 'color': "#fef9c3"},
                        {'range': [70, 100], 'color': "#fee2e2"}
                    ]
                }
            ))
            fig.update_layout(
                height=320, 
                margin=dict(l=30, r=30, t=50, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)

            # KEPUTUSAN & KATEGORI (1 Baris Rapi)
            if prob > 0.5:
                st.error(f"❌ **Pengajuan Pinjaman DITOLAK** &nbsp; | &nbsp; Kategori: **Risiko Tinggi**")
            else:
                st.success(f"✅ **Pengajuan Pinjaman DISETUJUI** &nbsp; | &nbsp; Kategori: **Risiko Rendah**")

            # Penjelasan Faktor Risiko
            st.markdown("<div class='section-header'>🧠 Faktor Risiko yang Mempengaruhi:</div>", unsafe_allow_html=True)
            
            factors_list = []
            if grade >= 5:
                factors_list.append("Grade kredit tinggi menunjukkan profil risiko historis yang lebih besar.")
            if dti > 20:
                factors_list.append("Rasio utang terhadap pendapatan cukup tinggi, mengurangi kapasitas bayar.")
            if inq_last_6mths > 2:
                factors_list.append("Terdapat banyak pengajuan kredit baru-baru ini, indikasi tekanan finansial.")
            if term == "60 months":
                factors_list.append("Tenor 60 bulan meningkatkan eksposur risiko terhadap perubahan kondisi ekonomi.")
            
            if factors_list:
                for factor in factors_list:
                    st.write(f"• {factor}")
            else:
                st.write("• Profil finansial peminjam tergolong stabil dan terkendali.")