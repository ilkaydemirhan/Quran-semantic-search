import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Sayfa yapılandırması
st.set_page_config(
    page_title="Kur'an Semantik/Anlamsal Arama Motoru", page_icon="📖", layout="centered"
)


# Verileri ve modeli önbelleğe alıyoruz
@st.cache_resource
def verileri_ve_modeli_yukle():
  # Saf Arapça ayetler için temiz kaynak (ar.asad veya ar.muyassar) ve Celaleyn
  url_arapca_saf = "http://api.alquran.cloud/v1/quran/ar.asad"
  url_celaleyn = "http://api.alquran.cloud/v1/quran/ar.jalalayn"
  url_turkce = "http://api.alquran.cloud/v1/quran/tr.diyanet"

  resp_tr = requests.get(url_turkce).json()
  resp_ar = requests.get(url_arapca_saf).json()
  resp_cel = requests.get(url_celaleyn).json()

  tum_ayetler = []
  for i, sure_tr in enumerate(resp_tr["data"]["surahs"]):
    sure_ar = resp_ar["data"]["surahs"][i]
    sure_cel = resp_cel["data"]["surahs"][i]
    sure_adi = sure_tr["englishName"]
    sure_islemeli = sure_tr["name"]

    for j, ayet_tr in enumerate(sure_tr["ayahs"]):
      ayet_saf = sure_ar["ayahs"][j]["text"]
      ayet_celaleyn = sure_cel["ayahs"][j]["text"]

      tum_ayetler.append({
          "sure": sure_islemeli,
          "sure_en": sure_adi,
          "ayet_no": ayet_tr["numberInSurah"],
          "arapca_ayet": ayet_saf,  # Saf Ayet Metni
          "celaleyn_tefsir": ayet_celaleyn,  # Tefsir Açıklaması
          "meal": ayet_tr["text"],  # Türkçe Diyanet Meali
      })

  df = pd.DataFrame(tum_ayetler)

  # Modeli yükle ve vektörleri hesapla
  model = SentenceTransformer(
      "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
  )
  # Arama yapılırken hem tefsir, hem saf ayet hem de meal baz alınır
  metinler = (
      df["arapca_ayet"] + " - " + df["celaleyn_tefsir"] + " - " + df["meal"]
  ).tolist()
  vektorler = model.encode(metinler)

  return df, model, vektorler


with st.spinner(
    "Veriler ve yapay zeka modeli yükleniyor, lütfen bekleyin..."
):
  df_kuran, model, vektorler = verileri_ve_modeli_yukle()

# Arayüz Tasarımı
st.title("📖 Kur'an Anlamsal Arama Motoru")
st.markdown(
    "Yapay zeka destekli bu arama motoru ile semantik/anlamsal"
    " sorgulama yapabilirsiniz."
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
      # 1. Adım: Yapay Zeka Benzerlikleri
      sorgu_vektoru = model.encode([sorgu])
      benzerlikler = cosine_similarity(sorgu_vektoru, vektorler)[0].copy()

      # 2. Adım: Hibrit Arama (Kelime Eşleşmesi)
      aranan_kelimeler = sorgu.lower().strip().split()
      for i, row in df_kuran.iterrows():
        meal_metni = str(row["meal"]).lower()
        for kelime in aranan_kelimeler:
          if len(kelime) > 2 and kelime in meal_metni:
            benzerlikler[i] += 0.15

      # 3. Adım: Sıralama
      en_iyi_indexler = np.argsort(benzerlikler)[::-1][: int(kac_adet)]

      st.markdown("### Sonuçlar")
      for sira, idx in enumerate(en_iyi_indexler, 1):
        ayet = df_kuran.iloc[idx]
        skor = benzerlikler[idx]

        with st.container():
          st.markdown(f"**{sira}. Sonuç** *(Arama Skoru: {skor:.2f})*")
          st.markdown(
              f"**Sure:** {ayet['sure']} ({ayet['sure_en']}) - Ayet"
              f" {ayet['ayet_no']}"
          )

          # 1. Saf Ayet Metni (Örn: Mavi / Info Kutusu)
          st.info(f"**Arapça Ayet:**\n\n{ayet['arapca_ayet']}")

          # 2. Celaleyn Tefsiri (Örn: Gri / Düz Metin veya Farklı Biçim)
          st.markdown(
              f"📝 **Celaleyn Tefsiri:**\n\n> {ayet['celaleyn_tefsir']}"
          )

          # 3. Türkçe Diyanet Meali (Örn: Yeşil / Success Kutusu)
          st.success(f"**Türkçe Diyanet Meali:**\n\n{ayet['meal']}")

          st.divider()
