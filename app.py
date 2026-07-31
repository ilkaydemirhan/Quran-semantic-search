import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Sayfa yapılandırması
st.set_page_config(
    page_title="Kur'an Anlamsal Arama Motoru", page_icon="📖", layout="centered"
)


# Verileri ve modeli önbelleğe alıyoruz ki her aramada baştan yüklemesin
@st.cache_resource
def verileri_ve_modeli_yukle():
  # Verileri çekme
  url_arapca = "http://api.alquran.cloud/v1/quran/ar.jalalayn"
  url_turkce = "http://api.alquran.cloud/v1/quran/tr.diyanet"
  resp_tr = requests.get(url_turkce).json()
  resp_ar = requests.get(url_arapca).json()

  tum_ayetler = []
  for i, sure_tr in enumerate(resp_tr["data"]["surahs"]):
    sure_ar = resp_ar["data"]["surahs"][i]
    sure_adi = sure_tr["englishName"]
    sure_islemeli = sure_tr["name"]
    for j, ayet_tr in enumerate(sure_tr["ayahs"]):
      ayet_ar = sure_ar["ayahs"][j]
      tum_ayetler.append({
          "sure": sure_islemeli,
          "sure_en": sure_adi,
          "ayet_no": ayet_tr["numberInSurah"],
          "arapca": ayet_ar["text"],
          "meal": ayet_tr["text"],
      })

  df = pd.DataFrame(tum_ayetler)

  # Modeli yükle ve vektörleri hesapla
  model = SentenceTransformer(
      "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  )
  metinler = (df["arapca"] + " - " + df["meal"]).tolist()
  vektorler = model.encode(metinler)

  return df, model, vektorler


with st.spinner(
    "Veriler ve yapay zeka modeli yükleniyor, lütfen bekleyin..."
):
  df_kuran, model, vektorler = verileri_ve_modeli_yukle()

# Arayüz Tasarımı
st.title("📖 Kur'an Anlamsal Arama Motoru")
st.markdown(
    "Yapay zeka destekli bu arama motoru ile kelimeler birebir tutmasa bile"
    " ayetlerin **anlamını** aratabilirsiniz."
)

# Arama girdileri
sorgu = st.text_input(
    "Arama Sorgusu",
    placeholder="Örn: merhamet, sabır, zorluk ve kolaylık...",
)
kac_adet = st.slider("Gösterilecek Sonuç Sayısı", min_value=1, max_value=20, value=5)

if st.button("Ara", type="primary"):
  if not sorgu.strip():
    st.warning("Lütfen geçerli bir arama terimi girin.")
  else:
    with st.spinner("Aranıyor..."):
      sorgu_vektoru = model.encode([sorgu])
      benzerlikler = cosine_similarity(sorgu_vektoru, vektorler)[0]
      en_iyi_indexler = np.argsort(benzerlikler)[::-1][: int(kac_adet)]

      st.markdown("### Sonuçlar")
      for sira, idx in enumerate(en_iyi_indexler, 1):
        ayet = df_kuran.iloc[idx]
        skor = benzerlikler[idx]

        with st.container():
          st.markdown(f"**{sira}. Sonuç** *(Benzerlik Skoru: {skor:.2f})*")
          st.markdown(
              f"**Sure:** {ayet['sure']} ({ayet['sure_en']}) - Ayet"
              f" {ayet['ayet_no']}"
          )
          st.info(f"**Arapça:** {ayet['arapca']}")
          st.success(f"**Meal:** {ayet['meal']}")
          st.divider()
