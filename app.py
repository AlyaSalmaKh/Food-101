import streamlit as st
import tensorflow as tf
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image
import time

# Konfigurasi halaman
st.set_page_config(
    page_title="🍴 Food-101 Classifier",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS custom untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .stProgress > div > div > div > div {
        background-color: #FF4B4B;
    }
    .prediction-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .food-emoji {
        font-size: 2rem;
        margin-right: 10px;
    }
    .confidence-bar {
        height: 20px;
        background-color: #e6e6e6;
        border-radius: 10px;
        margin: 10px 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background-color: #FF4B4B;
        border-radius: 10px;
        transition: width 0.5s ease-in-out;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar (Sederhana)
with st.sidebar:
    st.title("🍕 Food-101 Classifier")
    st.markdown("---")
    
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Gunakan gambar dengan pencahayaan baik
    - Fokus pada satu jenis makanan
    - Hindari gambar dengan banyak makanan
    - Format JPG/PNG dengan ukuran wajar
    """)

# Header utama
st.markdown('<h1 class="main-header">🍽️ Food-101 Image Classifier</h1>', unsafe_allow_html=True)
st.markdown("Upload gambar makanan dan model AI kami akan memprediksi jenis makanannya!")

# Load model dengan caching
@st.cache_resource
def load_model():
    try:
        # Repo Hugging Face
        REPO_ID = "Alya83/Food-101"
        MODEL_FILENAME = "model_resnet (1).h5"
        
        # Download model dari Hugging Face
        with st.spinner('🔄 Memuat model dari Hugging Face...'):
            model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
            model = tf.keras.models.load_model(model_path)
        
        st.success('✅ Model berhasil dimuat!')
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        return None

# Daftar kelas Food-101
class_names = [
    'apple_pie', 'baby_back_ribs', 'baklava', 'beef_carpaccio', 'beef_tartare',
    'beet_salad', 'beignets', 'bibimbap', 'bread_pudding', 'breakfast_burrito',
    'bruschetta', 'caesar_salad', 'cannoli', 'caprese_salad', 'carrot_cake',
    'ceviche', 'cheesecake', 'cheese_plate', 'chicken_curry', 'chicken_quesadilla',
    'chicken_wings', 'chocolate_cake', 'chocolate_mousse', 'churros', 'clam_chowder',
    'club_sandwich', 'crab_cakes', 'creme_brulee', 'croque_madame', 'cup_cakes',
    'deviled_eggs', 'donuts', 'dumplings', 'edamame', 'eggs_benedict', 'escargots',
    'falafel', 'filet_mignon', 'fish_and_chips', 'foie_gras', 'french_fries',
    'french_onion_soup', 'french_toast', 'fried_calamari', 'fried_rice',
    'frozen_yogurt', 'garlic_bread', 'gnocchi', 'greek_salad', 'grilled_cheese_sandwich',
    'grilled_salmon', 'guacamole', 'gyoza', 'hamburger', 'hot_and_sour_soup',
    'hot_dog', 'huevos_rancheros', 'hummus', 'ice_cream', 'lasagna', 'lobster_bisque',
    'lobster_roll_sandwich', 'macaroni_and_cheese', 'macarons', 'miso_soup',
    'mussels', 'nachos', 'omelette', 'onion_rings', 'oysters', 'pad_thai', 'paella',
    'pancakes', 'panna_cotta', 'peking_duck', 'pho', 'pizza', 'pork_chop',
    'poutine', 'prime_rib', 'pulled_pork_sandwich', 'ramen', 'ravioli', 'red_velvet_cake',
    'risotto', 'samosa', 'sashimi', 'scallops', 'seaweed_salad', 'shrimp_and_grits',
    'spaghetti_bolognese', 'spaghetti_carbonara', 'spring_rolls', 'steak',
    'strawberry_shortcake', 'sushi', 'tacos', 'takoyaki', 'tiramisu',
    'tuna_tartare', 'waffles'
]

# Fungsi untuk mendapatkan emoji berdasarkan kategori makanan
def get_food_emoji(food_name):
    emoji_map = {
        'pizza': '🍕', 'hamburger': '🍔', 'sushi': '🍣', 'steak': '🥩', 'ramen': '🍜',
        'ice_cream': '🍨', 'cheesecake': '🍰', 'chocolate_cake': '🎂', 'donuts': '🍩', 
        'apple_pie': '🥧', 'fried_rice': '🍚', 'spaghetti_bolognese': '🍝', 
        'chicken_curry': '🍛', 'tacos': '🌮', 'grilled_salmon': '🐟', 'waffles': '🧇',
        'pancakes': '🥞', 'bread_pudding': '🍮', 'cup_cakes': '🧁', 'hot_dog': '🌭'
    }
    
    for key, emoji in emoji_map.items():
        if key in food_name:
            return emoji
    return '🍽️'

# Load model
model = load_model()

# Upload Gambar
st.subheader("📤 Upload Gambar Makanan")
uploaded_file = st.file_uploader("Pilih file gambar", type=["jpg", "jpeg", "png"], key="uploader")

if uploaded_file is not None:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Tampilkan gambar
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Gambar yang diupload", use_container_width=True)
    
    with col2:
        if model is not None:
            # Preprocessing
            with st.spinner('Memproses gambar...'):
                img = image.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
            
            # Prediksi
            with st.spinner('Menganalisis gambar...'):
                time.sleep(1)  # Simulasi proses
                preds = model.predict(img_array, verbose=0)
                pred_probs = tf.nn.softmax(preds[0]).numpy()
            
            # Ambil top-5 prediksi
            top5_idx = pred_probs.argsort()[-5:][::-1]
            top5_labels = [class_names[i] for i in top5_idx]
            top5_scores = [pred_probs[i] for i in top5_idx]
            
            # Tampilkan hasil prediksi utama
            st.subheader("🔮 Hasil Prediksi")
            
            # Format nama makanan
            main_pred = top5_labels[0].replace('_', ' ').title()
            main_emoji = get_food_emoji(top5_labels[0])
            
            # Tampilkan prediksi utama dengan styling
            st.markdown(f"""
            <div class="prediction-card">
                <h3><span class="food-emoji">{main_emoji}</span> {main_pred}</h3>
                <p>Confidence: <strong>{top5_scores[0]*100:.2f}%</strong></p>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: {top5_scores[0]*100}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Tampilkan top-5 prediksi
            st.subheader("📊 Top-5 Prediksi")
            
            for i, (label, score) in enumerate(zip(top5_labels, top5_scores)):
                emoji = get_food_emoji(label)
                display_name = label.replace('_', ' ').title()
                confidence_percent = score * 100
                
                col_a, col_b = st.columns([2, 8])
                with col_a:
                    st.write(f"**{i+1}. {emoji} {display_name}**")
                with col_b:
                    st.progress(float(score), text=f"{confidence_percent:.1f}%")
            
            # Tampilkan detail prediksi
            with st.expander("🔍 Lihat Detail Prediksi"):
                for i, (label, score) in enumerate(zip(top5_labels, top5_scores)):
                    emoji = get_food_emoji(label)
                    st.write(f"{i+1}. {emoji} {label.replace('_', ' ').title()} - {score*100:.2f}%")
        else:
            st.error("Model tidak tersedia. Silakan coba lagi nanti.")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>Dibuat dengan menggunakan Streamlit | Model oleh Alya83</p>
</div>
""", unsafe_allow_html=True)