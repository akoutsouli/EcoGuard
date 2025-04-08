import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import random

# --------------------------
# Set Page Configuration
# --------------------------
st.set_page_config(page_title="EcoGuard AI", layout="centered")

# --------------------------
# Custom Nature-Themed CSS
# --------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #e8f5e9 0%, #c8e6c9 50%, #a5d6a7 100%);
        color: #1b5e20;
        font-family: 'Arial', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2e7d32;
        text-align: center;
        font-family: 'Georgia', serif;
    }
    p, label, .stMarkdown {
        color: #2e7d32;
    }
    div.stButton > button {
        background-color: #4caf50;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
        transition: background-color 0.3s ease-in-out;
    }
    div.stButton > button:hover {
        background-color: #388e3c;
    }
    .stRadio > label {
        color: #1b5e20;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True
)

# --------------------------
# Header & Sidebar
# --------------------------
st.title("🌿 EcoGuard AI 🌍")
st.header("Ένα app, ένας στόχος: ένας καθαρότερος κόσμος! ♻🌳")
st.write("Καλωσορίσατε στην EcoGuard AI! Επιλέξτε την επιθυμητή λειτουργία από την αριστερή πλευρά.")

st.sidebar.image("logo.png", use_container_width=True)
section = st.sidebar.radio("Επιλογή Λειτουργίας", ("Ανίχνευση Απόβλητων 🗑️", "Κουίζ Ανακύκλωσης 📝"))

# Define Waste Categories
recyclable_set = {"plastic", "paper", "metal", "glass", "cardboard", "bottle", "can"}
non_recyclable_set = {"organic", "hazardous", "styrofoam", "food waste", "battery", "diaper"}

# --------------------------
# Load CLIP model and processor
# --------------------------
@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

model, processor = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --------------------------
# Section 1: Waste Detection
# --------------------------
if section == "Ανίχνευση Απόβλητων 🗑️":
    st.subheader("Ανίχνευση Απόβλητων με CLIP 🖼️")
    st.write("Παρακαλώ ανεβάστε μια εικόνα για ανάλυση. 📸")
    
    uploaded_file = st.file_uploader("Επιλέξτε μια εικόνα", type=["png", "jpg", "jpeg"])
    image = None
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
    else:
        st.warning("Παρακαλώ ανεβάστε μια εικόνα για ανάλυση.")

    if image:
        st.image(image, caption="Επιλεγμένη Εικόνα", use_container_width=True)
        candidate_texts = list(recyclable_set | non_recyclable_set)
        inputs = processor(text=candidate_texts, images=image, return_tensors="pt", padding=True).to(device)
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        best_idx = probs.argmax().item()
        best_label = candidate_texts[best_idx]
        category = "Ανακυκλώσιμο ♻️" if best_label in recyclable_set else "Μη ανακυκλώσιμο 🚫"
        st.success(f"Ανιχνεύθηκε: **{best_label}**")
        st.info(f"Κατηγορία: **{category}**")
        st.caption(f"Πιθανότητα: {probs[0][best_idx]:.2%}")

# --------------------------
# Section 2: Recycling Quiz
# --------------------------
elif section == "Κουίζ Ανακύκλωσης 📝":
    st.subheader("Κουίζ Ανακύκλωσης 📝")
    st.write("Δοκιμάστε τις γνώσεις σας για την ανακύκλωση! 🌍")

    questions = [
        {"question": "Ποιο από τα παρακάτω είναι ανακυκλώσιμο;", 
         "options": ["Χαρτί 📄", "Φαγητά 🍲", "Μπαταρίες 🔋", "Οργανικά απόβλητα 🥕"], 
         "answer": "Χαρτί 📄"},
        {"question": "Πού πρέπει να πετάμε τις πλαστικές φιάλες;", 
         "options": ["Κάδος Ανακύκλωσης ♻️", "Κάδος Οργανικών 🥕"], 
         "answer": "Κάδος Ανακύκλωσης ♻️"},
        {"question": "Ποιο από τα παρακάτω υλικά χρειάζεται ειδική διαχείριση;", 
         "options": ["Πλαστικό 🥤", "Γυαλί 🍷", "Μπαταρίες 🔋", "Χαρτόνι 📦"], 
         "answer": "Μπαταρίες 🔋"},
        {"question": "Ποιο από τα παρακάτω υλικά μπορεί να ανακυκλωθεί ξανά και ξανά χωρίς να χάνει την ποιότητά του;", 
         "options": ["Αλουμίνιο 🥫", "Πλαστικό 🥤", "Χαρτί 📄", "Οργανικά απόβλητα 🥕"], 
         "answer": "Αλουμίνιο 🥫"},
        {"question": "Ποιο από τα παρακάτω υλικά θεωρείται ιδανικό για ανακύκλωση λόγω υψηλής αξίας;", 
         "options": ["Χαρτί 📄", "Αλουμίνιο 🥫", "Πλαστικό 🥤", "Γυαλί 🍷"], 
         "answer": "Αλουμίνιο 🥫"},
        {"question": "Γιατί είναι σημαντική η σωστή ταξινόμηση των απορριμμάτων;", 
         "options": ["Μειώνει τη ρύπανση", "Βελτιώνει την ανακύκλωση", "Ενισχύει τη βιωσιμότητα", "Όλα τα παραπάνω"], 
         "answer": "Όλα τα παραπάνω"}
    ]
    
    user_answers = {}
    for idx, q in enumerate(questions):
        st.markdown(f"**Ερώτηση {idx+1}:** {q['question']}")
        user_answers[idx] = st.radio("Επιλέξτε:", q["options"], key=f"quiz_{idx}")
        st.write("---")
    
    if st.button("Υποβολή Κουίζ 📤"):
        score = sum(1 for idx, q in enumerate(questions) if user_answers[idx] == q["answer"])
        st.success(f"Το σκορ σας: {score} / {len(questions)}")
        st.write("**Σωστές Απαντήσεις:**")
        for idx, q in enumerate(questions):
            st.write(f"Ερώτηση {idx+1}: {q['answer']}")

# --------------------------
# Link to External Game
# --------------------------
st.markdown(
    '''
    <div style="text-align:center; margin-top:20px;">
        <a href="https://akoutsouli.github.io/EcoBreaker/" target="_blank">
            <button style="padding:10px 20px; font-size:16px;">🎮 Παίξε το Παιχνίδι</button>
        </a>
    </div>
    ''',
    unsafe_allow_html=True
)
