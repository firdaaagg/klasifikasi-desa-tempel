"""Aplikasi Streamlit: Klasifikasi Informasi Portal Desa Tempel.

Model: TF-IDF + Logistic Regression (hasil perbandingan 4 model, lihat tab Performa Model).
Jalankan lokal:  streamlit run streamlit_app.py
"""
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KATEGORI_WARNA = {"Berita": "#c0392b", "Kegiatan": "#2e86c1", "Pengumuman": "#d68910", "Potensi Desa": "#239b56"}
AMBANG_TINJAU = 0.4  # sama dengan AI_REVIEW_THRESHOLD di backend website
MAX_JUDUL, MAX_KONTEN = 150, 100_000

st.set_page_config(page_title="Klasifikasi Informasi Desa Tempel", page_icon="🏘️", layout="wide")


def bersihkan_teks(teks) -> str:
    """Harus identik dengan fungsi saat pelatihan (train_model.py)."""
    if not isinstance(teks, str):
        return ""
    teks = teks.lower()
    teks = re.sub(r"<[^>]+>", " ", teks)
    teks = re.sub(r"[^a-z\s]", " ", teks)
    return re.sub(r"\s+", " ", teks).strip()


@st.cache_resource
def muat_model():
    return joblib.load(os.path.join(BASE_DIR, "model_klasifikasi_desa.joblib"))


@st.cache_data
def muat_data():
    return pd.read_csv(os.path.join(BASE_DIR, "dataset_desa_tempel.csv"))


@st.cache_data
def muat_metrik():
    with open(os.path.join(BASE_DIR, "metrik_evaluasi.json"), encoding="utf-8") as f:
        return json.load(f)


def klasifikasi(judul: str, konten: str, ringkasan: str = "") -> dict:
    model = muat_model()["model"]
    teks = bersihkan_teks(f"{judul} {judul} {ringkasan} {konten}")
    if not teks:
        raise ValueError("Teks tidak mengandung huruf yang dapat dianalisis.")
    proba = model.predict_proba([teks])[0]
    skor = sorted(({"kategori": str(k), "confidence": float(p)} for k, p in zip(model.classes_, proba)),
                  key=lambda s: s["confidence"], reverse=True)
    return {"kategori": skor[0]["kategori"], "confidence": skor[0]["confidence"], "skor": skor,
            "perlu_ditinjau": skor[0]["confidence"] < AMBANG_TINJAU}


@st.cache_data
def kata_penciri(n: int = 8) -> pd.DataFrame:
    """Kata dengan bobot terbesar per kategori (koefisien Logistic Regression)."""
    pipe = muat_model()["model"]
    tfidf, clf = pipe.named_steps["tfidf"], pipe.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return pd.DataFrame()
    kata = np.array(tfidf.get_feature_names_out())
    kolom = {}
    for i, k in enumerate(clf.classes_):
        kolom[str(k)] = list(kata[np.argsort(clf.coef_[i])[::-1][:n]])
    return pd.DataFrame(kolom, index=range(1, n + 1))


# ------------------------------------------------------------------ Header
st.title("🏘️ Klasifikasi Informasi Portal Desa Tempel")
st.caption("Asisten AI yang menyarankan kategori untuk informasi baru (Berita, Kegiatan, Pengumuman, Potensi Desa). "
           "Admin desa tetap yang memutuskan kategori akhir.")

tab_klasifikasi, tab_massal, tab_data, tab_model, tab_tentang = st.tabs(
    ["🔎 Klasifikasi", "📄 Klasifikasi Massal (CSV)", "📊 Eksplorasi Data", "🧠 Performa Model", "ℹ️ Tentang Sistem"])

# ------------------------------------------------------------------ Tab 1
with tab_klasifikasi:
    st.subheader("Cek kategori sebuah informasi")
    judul = st.text_input("Judul", max_chars=MAX_JUDUL, placeholder="Tulis judul informasi di sini...")
    st.caption(f"{len(judul)} / {MAX_JUDUL} karakter")
    ringkasan = st.text_area("Ringkasan", height=100, max_chars=1000,
                             placeholder="Ringkasan singkat (opsional), tampil di kartu daftar berita...")
    konten = st.text_area("Konten", height=260, max_chars=MAX_KONTEN,
                          placeholder="Tempel isi berita atau pengumuman di sini...")

    if st.button("Minta rekomendasi kategori", type="primary"):
        if not judul.strip() or not konten.strip():
            st.warning("Judul dan konten wajib diisi.")
        else:
            try:
                hasil = klasifikasi(judul.strip(), konten.strip(), ringkasan.strip())
            except ValueError as e:
                st.error(str(e))
            else:
                st.session_state["hasil"] = hasil

    hasil = st.session_state.get("hasil")
    if hasil:
        kol1, kol2 = st.columns([1, 1])
        with kol1:
            warna = KATEGORI_WARNA.get(hasil["kategori"], "#1f3a68")
            st.markdown(f"<div style='padding:16px;border-radius:10px;background:{warna}1a;border-left:6px solid {warna}'>"
                        f"<div style='font-size:0.85rem;color:#555'>Rekomendasi kategori</div>"
                        f"<div style='font-size:1.8rem;font-weight:700;color:{warna}'>{hasil['kategori']}</div>"
                        f"<div>Tingkat keyakinan: <b>{hasil['confidence']*100:.1f}%</b></div></div>", unsafe_allow_html=True)
            if hasil["perlu_ditinjau"]:
                st.warning(f"Keyakinan model di bawah {AMBANG_TINJAU*100:.0f}%. **Perlu ditinjau admin** sebelum kategori dipilih.")
            else:
                st.success("Keyakinan model cukup. Admin tetap dapat mengubah kategori bila perlu.")
        with kol2:
            skor_df = pd.DataFrame(hasil["skor"]).set_index("kategori")
            st.caption("Peluang tiap kategori")
            st.bar_chart(skor_df["confidence"], horizontal=True)

# ------------------------------------------------------------------ Tab 2
with tab_massal:
    st.subheader("Klasifikasi banyak informasi sekaligus")
    st.write("Unggah file CSV dengan kolom **judul** dan **konten** (kolom `ringkasan` opsional).")
    contoh_csv = pd.DataFrame({"judul": ["Judul informasi 1", "Judul informasi 2"],
                               "ringkasan": ["Ringkasan singkat 1", "Ringkasan singkat 2"],
                               "konten": ["Isi informasi 1 ...", "Isi informasi 2 ..."]})
    st.download_button("Unduh templat CSV", contoh_csv.to_csv(index=False).encode("utf-8"),
                       file_name="contoh_klasifikasi.csv", mime="text/csv")
    berkas = st.file_uploader("Pilih file CSV", type=["csv"])
    if berkas is not None:
        try:
            data = pd.read_csv(berkas)
        except Exception:
            st.error("File tidak dapat dibaca sebagai CSV.")
            data = None
        if data is not None:
            if not {"judul", "konten"}.issubset(data.columns):
                st.error("Kolom `judul` dan `konten` wajib ada.")
            elif len(data) > 500:
                st.error("Maksimal 500 baris per unggahan.")
            else:
                baris = []
                for _, r in data.iterrows():
                    try:
                        h = klasifikasi(str(r["judul"]), str(r["konten"]), str(r.get("ringkasan", "") if pd.notna(r.get("ringkasan", "")) else ""))
                        baris.append({"judul": r["judul"], "kategori": h["kategori"],
                                      "confidence": round(h["confidence"], 3), "perlu_ditinjau": h["perlu_ditinjau"]})
                    except ValueError:
                        baris.append({"judul": r["judul"], "kategori": "-", "confidence": 0.0, "perlu_ditinjau": True})
                hasil_df = pd.DataFrame(baris)
                st.dataframe(hasil_df, width="stretch", hide_index=True)
                st.caption(f"{int(hasil_df['perlu_ditinjau'].sum())} dari {len(hasil_df)} informasi ditandai perlu ditinjau.")
                st.download_button("Unduh hasil (CSV)", hasil_df.to_csv(index=False).encode("utf-8"),
                                   file_name="hasil_klasifikasi.csv", mime="text/csv")

# ------------------------------------------------------------------ Tab 3
with tab_data:
    df = muat_data()
    st.subheader("Memahami data latih")
    a, b, c = st.columns(3)
    a.metric("Jumlah data", len(df))
    b.metric("Jumlah kategori", df["kategori"].nunique())
    asli = int((df["sumber"] == "asli").sum()) if "sumber" in df else len(df)
    c.metric("Data asli / tambahan", f"{asli} / {len(df) - asli}")
    st.info("Sebagian data ditulis sebagai data ilustratif (kolom `sumber = tambahan`). "
            "Karena itu angka performa belum tentu berlaku pada berita nyata.")

    kiri, kanan = st.columns(2)
    with kiri:
        st.caption("Distribusi data per kategori dan sumber")
        if "sumber" in df:
            st.bar_chart(df.groupby(["kategori", "sumber"]).size().unstack(fill_value=0))
        else:
            st.bar_chart(df["kategori"].value_counts())
    with kanan:
        st.caption("Rata-rata panjang isi (jumlah kata) per kategori")
        df["jumlah_kata"] = df["konten"].fillna("").str.split().str.len()
        st.bar_chart(df.groupby("kategori")["jumlah_kata"].mean().round(0))

    st.caption("Kata penciri tiap kategori (bobot terbesar pada model)")
    penciri = kata_penciri()
    if not penciri.empty:
        st.dataframe(penciri, width="stretch")

    st.caption("Cuplikan dataset")
    saring = st.multiselect("Saring kategori", sorted(df["kategori"].unique()), default=[])
    tampil = df[df["kategori"].isin(saring)] if saring else df
    st.dataframe(tampil[["judul", "kategori"] + (["sumber"] if "sumber" in df else [])], width="stretch", hide_index=True)

# ------------------------------------------------------------------ Tab 4
with tab_model:
    m = muat_metrik()
    st.subheader("Perbandingan 4 model")
    st.write("Setiap model menyetel parameternya sendiri lalu diuji dengan *nested cross-validation* "
             "(5-fold x 2 pengulangan), sehingga hasilnya tidak terlalu optimis.")
    perb = pd.DataFrame(m["perbandingan_model_nested_cv"]).T.rename(columns={
        "akurasi_mean": "Akurasi", "akurasi_std": "Std akurasi", "f1_macro_mean": "F1-macro",
        "f1_macro_std": "Std F1", "waktu_detik": "Waktu (dtk)"}).sort_values("F1-macro", ascending=False)
    st.dataframe(perb.round(3), width="stretch")
    st.image(os.path.join(BASE_DIR, "perbandingan_model.png"))
    st.success(f"Model terpilih: **{m['model_terpilih']}**. Selisih antarmodel kecil, sehingga dipilih yang skornya tertinggi, "
               "sederhana, dan probabilitasnya langsung dipakai sebagai tingkat keyakinan.")

    st.subheader("Evaluasi model terpilih")
    x, y, z = st.columns(3)
    x.metric("Akurasi CV 5-fold", f"{m['cv_5fold']['akurasi']*100:.1f}%")
    y.metric(f"Akurasi hold-out ({m['holdout_25persen']['jumlah_uji']} data)", f"{m['holdout_25persen']['akurasi']*100:.1f}%")
    if m.get("latih_tambahan_uji_asli_akurasi") is not None:
        z.metric("Latih tambahan, uji data asli", f"{m['latih_tambahan_uji_asli_akurasi']*100:.1f}%")
    st.image(os.path.join(BASE_DIR, "confusion_matrix_terbaik.png"), width=480)
    lap = pd.DataFrame(m["laporan_per_kelas"]).T.loc[["Berita", "Kegiatan", "Pengumuman", "Potensi Desa"],
                                                      ["precision", "recall", "f1-score", "support"]]
    st.dataframe(lap.round(3), width="stretch")
    st.caption("Keterbatasan: data hanya 72 baris dan sebagian ilustratif. Kategori Berita dan Kegiatan paling sering tertukar. "
               "Tingkat keyakinan adalah probabilitas model, bukan jaminan kebenaran.")

# ------------------------------------------------------------------ Tab 5
with tab_tentang:
    st.subheader("Cara kerja sistem")
    st.markdown("""
**Alur:** admin menulis judul dan isi informasi → teks dibersihkan (huruf kecil, buang tag HTML dan tanda baca) →
diubah menjadi vektor **TF-IDF** (unigram dan bigram) → **Logistic Regression** menghitung peluang tiap kategori →
sistem menampilkan kategori teratas, tingkat keyakinan, dan peluang kategori lain.

**Ambang "perlu ditinjau":** bila keyakinan di bawah 40%, informasi ditandai untuk diperiksa admin. Pada validasi silang,
semua prediksi yang salah berada di bawah ambang ini.

**Integrasi dengan website:** model yang sama dilayani lewat layanan FastAPI (`POST /classify`) yang dipanggil backend
Express pada tombol *Minta rekomendasi* di halaman admin Portal Informasi Desa Tempel. Aplikasi Streamlit ini adalah
antarmuka demonstrasi dan analisis dari model yang sama.
""")
    info = muat_model()
    st.caption(f"Model: {info.get('nama_model')} | Data latih: {info.get('jumlah_data_latih')} | scikit-learn {info.get('versi_sklearn', '-')}")
