import numpy as np
import pandas as pd
import requests
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Kur'an Semantik Arama", page_icon="📖", layout="wide")

@st.cache_data
def verileri_cek():
    """API'den verileri çeker ve DataFrame'i hazırlar."""
    # Veri kaynağı (tüm Kuran)
    url_arapca = "http://api.alquran.cloud/v1/quran/ar.asad"
    url_tr = "http://api.alquran.cloud/v1/quran/tr.diyanet"
    
    resp_ar = requests.get(url_arapca).json()
    resp_tr = requests.get(url_tr).json()
    
    data = []
    for i, sure_tr in enumerate(resp_tr["data"]["surahs"]):
        sure_ar = resp_ar["data"]["surahs"][i]
        for j, ayet_tr in enumerate(sure_tr["ayahs"]):
            data.append({
                "sure": sure_tr["name"],
                "sure_en": sure_tr["englishName"],
                "ayet_no": ayet_tr["numberInSurah"],
                "arapca": sure_ar["ayahs"][j]["text"],
                "meal": ayet_tr["text"]
            })
    return pd.DataFrame(data)

@st.cache_resource
def model_yukle():
    """Modeli bir kez yükler."""
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_data
def vektorleri_hesapla(_df):
    """Metinleri vektöre dönüştürür."""
    model = model_yukle()
    metinler = (_df["arapca"] + " - " + _df["meal"]).tolist()
    return model.encode(metinler)

# --- Uygulama Başlangıcı ---
df = verileri_cek()
model = model_yukle()
vektorler = vektorleri_hesapla(df)

st.title("📖 Kur'an Semantik Arama Motoru")

tab1, tab2 = st.tabs(["🔍 Semantik Arama", "ℹ️ Proje Hakkında"])

with tab1:
    sorgu = st.text_input("Arama Sorgusu:", placeholder="Örn: 'insanın yaratılışı', 'sabır ve tevekkül'...")
    kac_adet = st.slider("Sonuç Sayısı", 1, 20, 5)
    
    if st.button("Ara"):
        if sorgu:
            sorgu_vektoru = model.encode([sorgu])
            benzerlikler = cosine_similarity(sorgu_vektoru, vektorler)[0]
            en_iyi_idx = np.argsort(benzerlikler)[::-1][:kac_adet]
            
            for idx in en_iyi_idx:
                ayet = df.iloc[idx]
                with st.container():
                    st.markdown(f"**{ayet['sure']} ({ayet['sure_en']}) - {ayet['ayet_no']}. Ayet**")
                    st.info(f"*{ayet['arapca']}*")
                    st.success(ayet['meal'])
                    st.divider()
        else:
            st.warning("Lütfen bir arama terimi girin.")

with tab2:
    st.markdown("""
    ### Proje Detayları
    * **Model:** Multilingual MiniLM (Çok dilli anlamsal analiz)
    * **Veri:** Al-Quran Cloud API
    * **Özellik:** Arapça metin ve Türkçe meal üzerinde hibrit anlamsal arama.
    * **Geliştirme:** Bir sonraki aşamada kök bazlı morfolojik filtreleme eklenecektir.
    """)
