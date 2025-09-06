import streamlit as st
import tensorflow as tf
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

# Repo Hugging Face
REPO_ID = "Alya83/Food-101"
MODEL_FILENAME = "model_resnet (1).h5"

# Download model dari Hugging Face
model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
model = tf.keras.models.load_model(model_path)

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

# UI
st.set_page_config(page_title="🍴 Food-101 Classifier", layout="centered")

st.title("🍽️ Food-101 Image Classifier")
st.markdown("Upload gambar makanan dan model akan memprediksi jenis makanannya!")

uploaded_file = st.file_uploader("📤 Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Gambar yang diupload", use_column_width=True)

    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    preds = model.predict(img_array)
    pred_probs = tf.nn.softmax(preds[0]).numpy()

    # Ambil top-3 prediksi
    top3_idx = pred_probs.argsort()[-3:][::-1]
    top3_labels = [class_names[i] for i in top3_idx]
    top3_scores = [pred_probs[i] for i in top3_idx]

    st.subheader("🔮 Hasil Prediksi")
    st.success(f"Prediksi utama: **{top3_labels[0]}** 🍔 (Confidence: {top3_scores[0]*100:.2f}%)")

    # Tampilkan top-3 dalam metric cards
    st.write("### 📊 Top-3 Prediksi")
    col1, col2, col3 = st.columns(3)
    col1.metric("🥇", top3_labels[0], f"{top3_scores[0]*100:.1f}%")
    col2.metric("🥈", top3_labels[1], f"{top3_scores[1]*100:.1f}%")
    col3.metric("🥉", top3_labels[2], f"{top3_scores[2]*100:.1f}%")
