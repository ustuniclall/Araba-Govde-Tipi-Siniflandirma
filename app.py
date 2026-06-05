import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass
import logging
from PIL import Image
from torchvision.models import efficientnet_b0
import time

# SAYFA AYARI
st.set_page_config(
    page_title="🚘 AI Car Body Classifier",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# MODERN TASARIM
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to bottom, #0B2535, #123B52);
    color: white;
}
h1 {
    color: white !important;
    font-size: 50px !important;
    font-weight: 800 !important;
    text-align: center;
}
h2, h3 {
    color: #EAF6FF !important;
}
.stButton > button {
    background: linear-gradient(90deg, #ff2d2d, #ff5c5c);
    color: white;
    border: none;
    border-radius: 12px;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
    width: 100%;
}
.stButton > button:hover {
    transform: scale(1.02);
}
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.05);
    border-radius: 15px;
    padding: 15px;
}
section[data-testid="stSidebar"] {
    background: #102C3D;
}
[data-testid="stFileUploaderFileName"] {
    color: white !important;
    font-weight: bold !important;
}
</style>
""", unsafe_allow_html=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MODEL CONFIG
@dataclass
class ModelConfig:
    num_classes: int = 8
    image_size: Tuple[int, int] = (224, 224)
    model_path: str = "proje_model.pth"

# MODEL SINIFI
class CarClassifier:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Modelin eğitimdeki gerçek alfabetik çıkış sırası
        self.labels_model = [
            'f1', 'hatchback', 'micro', 'pickup', 'sedan', 'station_wagon', 'suv', 'van'
        ]
        
        # 2. Arayüz ekranında kullanıcıya gösterilecek Türkçe Gövde Tipleri
        self.labels_display = [
            "AÇIK TEKERLEKLİ (F1 ARAÇLARI)", "HATCHBACK", "MICRO", "PICK UP", 
            "SEDAN", "STATION WAGON", "SUV", "VAN"
        ]
        
        # 3. HOCANIN TEST SCRIPTININ BEKLEDİĞİ RESMİ ID KARŞILIKLARI (1-8 ARASI)
        # Sıralama yukardaki alfabetik modele göredir:
        # f1->5, hatchback->7, micro->4, pickup->8, sedan->6, station_wagon->3, suv->1, van->2
        self.hoca_ids = [5, 7, 4, 8, 6, 3, 1, 2]
        
        # 4. Grafik gösterimi için hocanın resmi isim sıralaması (1'den 8'e)
        self.hoca_sirali_isimler = [
            "1. SUV", "2. VAN", "3. STATION WAGON", "4. MICRO", 
            "5. AÇIK TEKERLEKLİ (F1)", "6. SEDAN", "7. HATCHBACK", "8. PICK UP"
        ]
        
        self._load_model()
        self._setup_transforms()

    def _load_model(self):
        self.model = efficientnet_b0(weights=None)
        self.model.classifier[1] = nn.Linear(1280, 8)
        
        checkpoint = torch.load(self.config.model_path, map_location=self.device)
        
        # Paket (Checkpoint) kontrolü
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            self.model.load_state_dict(checkpoint)
            
        self.model.to(self.device)
        self.model.eval()

    def _setup_transforms(self):
        from torchvision import transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def predict(self, image, file_name: str):
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        start_time = time.time()
        outputs = self.model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        inference_time = time.time() - start_time

        # En yüksek olasılıklı ham indeks (0-7 arası)
        predicted_idx = np.argmax(probabilities)
        
        # Doğru etiket adını ve hocanın beklediği resmi sayısal ID'yi çekiyoruz
        hocanin_display_name = self.labels_display[predicted_idx]
        hocanin_id = self.hoca_ids[predicted_idx]
        
        # ⚠️ HOCANIN "PredictionScript.txt" FORMATINA %100 UYGUN DOSYA YAZIMI ⚠️
        # Satır formatı kesinlikle: "dosya_adi.jpg | Class: X" şeklinde olmalıdır.
        with open("preds.txt", "a", encoding="utf-8") as f:
            f.write(f"{file_name} | Class: {hocanin_id}\n")

        # Grafik çubuklarının hocanın 1'den 8'e olan resmi şablon sırasına göre dizilmesi
        hoca_sirali_probs = [
            probabilities[self.labels_model.index('suv')] * 100,
            probabilities[self.labels_model.index('van')] * 100,
            probabilities[self.labels_model.index('station_wagon')] * 100,
            probabilities[self.labels_model.index('micro')] * 100,
            probabilities[self.labels_model.index('f1')] * 100,
            probabilities[self.labels_model.index('sedan')] * 100,
            probabilities[self.labels_model.index('hatchback')] * 100,
            probabilities[self.labels_model.index('pickup')] * 100
        ]

        return {
            "predicted_class": hocanin_display_name,
            "confidence": probabilities[predicted_idx] * 100,
            "inference_time": inference_time,
            "ordered_labels": self.hoca_sirali_isimler,
            "ordered_probs": hoca_sirali_probs,
            "hocanin_id": hocanin_id
        }

# MODEL YÜKLEME (Streamlit Önbellekli)
@st.cache_resource
def load_model():
    config = ModelConfig()
    return CarClassifier(config)

def main():
    st.markdown("<h1 style='text-align: center;'> 🚘 AI Car Body Classifier</h1>", unsafe_allow_html=True)
    st.write("---")

    classifier = load_model()

    col1, col2 = st.columns([1, 1], gap="large")

    # --- SOL SÜTUN: GÖRSEL YÜKLEME ---
    with col1:
        st.subheader("📤 Görsel Yükleme")
        uploaded_file = st.file_uploader(
            "Araba görseli seçin",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=f"Yüklenen Görsel: {uploaded_file.name}", use_container_width=True)

    # --- SAĞ SÜTUN: TAHMİN PANELİ ---
    with col2:
        st.subheader("🤖 Tahmin Paneli")
        predict_button = st.button("🚀 Tahmin Yap", use_container_width=True)

        if uploaded_file and predict_button:
            prediction = classifier.predict(image, uploaded_file.name)

            # Sonuç Ekranı Kutusu
            st.markdown(
                f"""
                <div style="
                    background-color:white;
                    padding:12px 14px;
                    border-radius:12px;
                    border:2px solid #ff4b4b;
                    font-size:22px;
                    font-weight:700;
                    color:#ff2b2b;
                    text-align:center;
                    box-shadow:0 4px 12px rgba(0,0,0,0.15);
                    margin-top:10px;
                    margin-bottom:10px;
                ">
                    🚘 {prediction['predicted_class']} (Sınıf Kod: {prediction['hocanin_id']})<br>
                    <span style='font-size:16px; color:#555;'>Güven Skoru: %{prediction['confidence']:.2f} | Süre: {prediction['inference_time']:.4f} sn</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Matplotlib Olasılık Grafiği (Hoca Düzeni)
            fig, ax = plt.subplots(figsize=(8, 5))
            
            fig.patch.set_facecolor('#102C3D') 
            ax.set_facecolor('#102C3D')
            ax.tick_params(colors='white', labelsize=10)
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

            bars = ax.bar(
                prediction["ordered_labels"],
                prediction["ordered_probs"],
                color="#ff3b3b",
                edgecolor="white",
                linewidth=0.7
            )

            ax.set_ylim(0, 115)
            ax.set_ylabel("Olasılık (%)", fontsize=11, fontweight='bold')
            ax.set_title("Sınıf Olasılık Dağılımı (Hoca Standartı)", pad=15, fontsize=12, fontweight='bold')

            # Çubukların üzerine yüzdeleri yazdırma
            for bar in bars:
                height = bar.get_height()
                if height > 1: 
                    ax.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 2,
                        f'%{height:.1f}',
                        ha='center', va='bottom', color='white', fontsize=9, fontweight='bold'
                    )

            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.success(f"Başarılı: '{uploaded_file.name} | Class: {prediction['hocanin_id']}' formatı preds.txt dosyasına eklendi.")
            
        elif not uploaded_file and predict_button:
            st.info("Lütfen analiz için bir görsel yükleyin.")

if __name__ == "__main__":
    main()