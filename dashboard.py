import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# --- Fungsi binning (SAMA dengan training pipeline) ---
def add_bins_one_row(age, credit, duration):
    age_bin = str(pd.cut([age], bins=[18,25,30,35,40,50,60,100])[0])
    credit_bin = str(pd.cut([credit], bins=[0,2000,5000,10000,1e10],
                        labels=['very low','low','mid','high'],
                        right=False, include_lowest=True)[0])
    duration_bin = pd.cut([duration], bins=[0, 12, 24, 36, 48, 100], 
                         labels=['<=12', '13-24', '25-36', '37-48', '>48'])[0]
    return age_bin, credit_bin, str(duration_bin)

# --- Load model ---
@st.cache_resource
def load_model():
    with open("model/model_prediksi-kredit.pkl", "rb") as f:
        return pickle.load(f)
model = load_model()

# --- Sidebar Input ---
st.sidebar.title("📝 Input Data Calon Nasabah")

default = {
    "Age": 35,
    "Sex": "male",
    "Job": 2,
    "Housing": "own",
    "Saving accounts": "little",
    "Checking account": "moderate",
    "Credit amount": 2000,
    "Duration": 12,
    "Purpose": "radio/TV"
}

with st.sidebar.form("input_form"):
    Age = st.number_input("Umur", 18, 100, default["Age"])
    Sex = st.selectbox("Jenis Kelamin", ["male", "female"], index=0)
    Job = st.number_input("Job (0=simple, 3=professional)", 0, 3, default["Job"])
    Housing = st.selectbox("Housing", ["own", "rent", "free"], index=0)
    Saving_accounts = st.selectbox("Saving accounts", ["little", "moderate", "quite rich","rich","Unknown"], index=1)
    Checking_account = st.selectbox("Checking account", ["little", "moderate", "rich", "Unknown"], index=2)
    Credit_amount = st.number_input("Credit Amount", 0, 50000, default["Credit amount"])
    Duration = st.number_input("Duration (bulan)", 1, 100, default["Duration"])
    Purpose = st.selectbox("Purpose", [
        "radio/TV","car","furniture/equipment","business",
        "education","vacation/others","repairs","domestic appliances"
    ], index=0)
    submit = st.form_submit_button("Prediksi Risiko Kredit")

st.title("📊 Dashboard Prediksi Risiko Kredit Nasabah")

if submit:
    age_bin, credit_bin, duration_bin = add_bins_one_row(Age, Credit_amount, Duration)

    input_df = pd.DataFrame([{
        "Age": Age,
        "Sex": Sex,
        "Job": Job,
        "Housing": Housing,
        "Saving accounts": Saving_accounts,
        "Checking account": Checking_account,
        "Credit amount": Credit_amount,
        "Duration": Duration,
        "Purpose": Purpose,
        "age_bin": age_bin,
        "credit_bin": credit_bin,
        "duration_bin": duration_bin
    }])

    # Transformasi fitur (sesuai pipeline)
    preprocessor = model.named_steps['transformer']
    X_trans = preprocessor.transform(input_df)
    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()
    rf = model.named_steps['model']

    proba_default = rf.predict_proba(X_trans)[:, 1][0]
    threshold = 0.54
    pred_label = int(proba_default >= threshold)
    risk_segment = ("Low" if proba_default < 0.4 else
                    "Medium" if proba_default < 0.7 else
                    "High")

    # Layout hasil prediksi
    st.subheader("Hasil Prediksi")
    col1, col2, col3 = st.columns(3)
    col1.metric("Probabilitas Gagal Bayar", f"{proba_default:.3f}")
    col2.metric("Segmen Risiko", risk_segment)
    col3.metric("Prediksi", "BAD" if pred_label else "GOOD")

    # Info tambahan
    st.write("---")
    st.write("#### Rincian Data Input")
    st.dataframe(input_df, use_container_width=True)

    # Feature Importances
    st.write("#### Feature Importances (Global, Random Forest)")
    importances = rf.feature_importances_
    feat_names = preprocessor.get_feature_names_out()
    imp_df = pd.DataFrame({
        'Feature': feat_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Ambil satu fitur paling penting
    top_feature = imp_df.iloc[0]
    top_feat_name = top_feature['Feature']
    top_feat_import = top_feature['Importance'] * 100

    # Tampilkan sebagai metric (dengan persen)
    st.write("#### Fitur Paling Penting")
    st.metric(
        label=f"Fitur: {top_feat_name}",
        value=f"{top_feat_import:.2f} %"
    )


    st.dataframe(imp_df.head(10), use_container_width=True)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(imp_df.head(10)['Feature'][::-1], imp_df.head(10)['Importance'][::-1])
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Feature Importances")
    plt.tight_layout()
    st.pyplot(fig)

    st.caption("Model: Random Forest + preprocessing pipeline. Prediksi hanya sebagai simulasi, bukan keputusan akhir.")
else:
    st.info("Silakan masukkan data nasabah di sidebar dan klik tombol **Prediksi Risiko Kredit**.")

