# --------------------------
# Set Page Configuration - MUST BE FIRST STREAMLIT COMMAND
# --------------------------
import streamlit as st

st.set_page_config(
    page_title="EcoGuard AI",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --------------------------
# Imports (after set_page_config)
# --------------------------
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import os

# Cache folder (helps on Streamlit Cloud)
os.environ["HF_HOME"] = "./hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "./hf_cache"

# --------------------------
# Load CLIP model ON DEMAND (only when needed)
# --------------------------
@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

# --------------------------
# Custom Nature-Themed CSS + Animations
# --------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Poppins:wght@300;600&display=swap');

    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 50%, #a5d6a7 100%);
        color: #1b5e20;
        font-family: 'Poppins', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #2e7d32;
        text-align: center;
        font-family: 'Montserrat', sans-serif;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    .stMarkdown p {
        font-size: 1.1rem;
    }

    div.stButton > button {
        background: linear-gradient(to right, #4CAF50, #2E7D32);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 12px 24px;
        font-weight: bold;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
        font-family: 'Montserrat', sans-serif;
    }

    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }

    .stRadio > div {
        background-color: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        padding: 15px;
    }

    .stSuccess, .stInfo {
        border-radius: 10px;
        padding: 15px;
    }

    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
        100% { transform: translateY(0px); }
    }

    .floating {
        animation: float 3s ease-in-out infinite;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------
# Header & Sidebar
# --------------------------
st.title("🌿 EcoGuard AI 🌍")
st.markdown(
    """
    <div style='text-align:center;'>
        <h3 style='color:#2e7d32;'>Ένα app, ένας στόχος: ένας καθαρότερος κόσμος! ♻️🌳</h3>
        <p style='font-size:1.2rem;'>Χρησιμοποιήστε τεχνητή νοημοσύνη για να μάθετε περισσότερα για την ανακύκλωση!</p>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown(
        """
        <div class='floating' style='text-align:center; margin-bottom:30px;'>
            <img src='https://cdn-icons-png.flaticon.com/512/3095/3095583.png' width='100'>
        </div>
        """,
        unsafe_allow_html=True
    )
    section = st.radio(
        "Επιλογή Λειτουργίας",
        ("Ανίχνευση Απόβλητων 🗑️", "Κουίζ Ανακύκλωσης 📝", "Παιχνίδι ♻️", "Eco-Tips 💡"),
        index=0
    )

# --------------------------
# Waste Categories
# --------------------------
recyclable_set = {
    "plastic": "♻️ Πλαστικό",
    "paper": "♻️ Χαρτί",
    "metal": "♻️ Μέταλλο",
    "glass": "♻️ Γυαλί",
    "cardboard": "♻️ Χαρτόνι",
    "bottle": "♻️ Φιάλη",
    "can": "♻️ Κονσέρβα"
}

non_recyclable_set = {
    "organic": "🚫 Οργανικά",
    "hazardous": "☢️ Επικίνδυνα",
    "styrofoam": "🚫 Στυροπιν",
    "food waste": "🚫 Τρόφιμα",
    "battery": "☢️ Μπαταρία",
    "diaper": "🚫 Πάνες"
}

# --------------------------
# Section 1: Waste Detection
# --------------------------
if section == "Ανίχνευση Απόβλητων 🗑️":
    st.subheader("🔍 Ανίχνευση Απόβλητων με AI")
    st.markdown(
        """
        <div style='background-color:rgba(255,255,255,0.7); padding:15px; border-radius:10px;'>
        <p>📸 Ανεβάστε μια φωτογραφία και το AI θα σας πει αν είναι ανακυκλώσιμο!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Επιλέξτε εικόνα",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Η εικόνα σας", use_container_width=True)

        with st.spinner("Φόρτωση μοντέλου AI... ⏳"):
            try:
                model, processor, device = load_clip()
            except Exception:
                st.error("Το μοντέλο δεν φορτώθηκε στο Streamlit Cloud (συνήθως θέμα σύνδεσης/περιβάλλοντος). Άνοιξε Manage app → Logs για τη λεπτομέρεια.")
                st.stop()

        with st.spinner("Ανάλυση εικόνας... 🔍"):
            candidate_labels = list(recyclable_set.keys()) + list(non_recyclable_set.keys())

            inputs = processor(
                text=candidate_labels,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(device)

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

            top_probs, top_indices = torch.topk(probs, 3)

            st.success("🔎 Αποτελέσματα Ανάλυσης")

            result_col1, result_col2 = st.columns(2)

            best_idx = top_indices[0][0].item()
            best_label = candidate_labels[best_idx]

            with result_col1:
                if best_label in recyclable_set:
                    st.markdown(
                        f"""
                        <div style='background-color:#E8F5E9; padding:15px; border-radius:10px;'>
                        <h3 style='color:#2E7D32; text-align:center;'>♻️ Ανακυκλώσιμο</h3>
                        <p style='text-align:center; font-size:1.3rem;'><b>{recyclable_set[best_label]}</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f"""
                        <div style='background-color:#FFEBEE; padding:15px; border-radius:10px;'>
                        <h3 style='color:#C62828; text-align:center;'>🚫 Μη Ανακυκλώσιμο</h3>
                        <p style='text-align:center; font-size:1.3rem;'><b>{non_recyclable_set[best_label]}</b></p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            with result_col2:
                st.markdown(
                    """
                    <div style='background-color:white; padding:15px; border-radius:10px;'>
                    <h4 style='color:#2E7D32;'>🔍 Άλλες πιθανότητες:</h4>
                    """,
                    unsafe_allow_html=True
                )

                for i in range(1, 3):
                    idx = top_indices[0][i].item()
                    label = candidate_labels[idx]
                    prob = top_probs[0][i].item()

                    disp_name = recyclable_set.get(label, non_recyclable_set.get(label, label))
                    st.markdown(
                        f"<p>- {disp_name} <span style='float:right;'>{prob:.1%}</span></p>",
                        unsafe_allow_html=True
                    )

                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("---")
            if best_label in recyclable_set:
                st.markdown(
                    """
                    <div style='background-color:#E8F5E9; padding:15px; border-radius:10px;'>
                    <h4 style='color:#2E7D32;'>✅ Πώς να ανακυκλώσετε:</h4>
                    <ul>
                        <li>Καθαρίστε το αντικείμενο από υπολείμματα</li>
                        <li>Αφαιρέστε πώματα ή καπάκια</li>
                        <li>Τοποθετήστε στον κατάλληλο κάδο ανακύκλωσης</li>
                    </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    """
                    <div style='background-color:#FFF3E0; padding:15px; border-radius:10px;'>
                    <h4 style='color:#E65100;'>⚠️ Σημαντική Σημείωση:</h4>
                    <p>Αυτό το αντικείμενο δεν πρέπει να ανακυκλωθεί. Αν είναι επικίνδυνο (π.χ. μπαταρίες), 
                    ψάξτε για ειδικά σημεία συλλογής στην περιοχή σας.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# --------------------------
# Section 2: Recycling Quiz
# --------------------------
elif section == "Κουίζ Ανακύκλωσης 📝":
    st.subheader("📚 Κουίζ Ανακύκλωσης")
    st.markdown(
        """
        <div style='background-color:rgba(255,255,255,0.7); padding:15px; border-radius:10px; margin-bottom:20px;'>
        <p>Δοκιμάστε τις γνώσεις σας με αυτό το διαδραστικό κουίζ!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    questions = [
        {
            "question": "Ποιο από τα παρακάτω είναι ανακυκλώσιμο;",
            "options": ["Χαρτί 📄", "Φαγητά 🍲", "Μπαταρίες 🔋", "Οργανικά απόβλητα 🥕"],
            "answer": "Χαρτί 📄",
            "explanation": "Το χαρτί μπορεί να ανακυκλωθεί πολλές φορές."
        },
        {
            "question": "Πού πρέπει να πετάμε τις πλαστικές φιάλες;",
            "options": ["Κάδος Ανακύκλωσης ♻️", "Κάδος Οργανικών 🥕"],
            "answer": "Κάδος Ανακύκλωσης ♻️",
            "explanation": "Οι φιάλες πάνε στον κάδο ανακύκλωσης."
        },
        {
            "question": "Ποιο χρειάζεται ειδική διαχείριση;",
            "options": ["Πλαστικό 🥤", "Γυαλί 🍷", "Μπαταρίες 🔋", "Χαρτόνι 📦"],
            "answer": "Μπαταρίες 🔋",
            "explanation": "Οι μπαταρίες έχουν επικίνδυνες ουσίες."
        },
        {
            "question": "Το αλουμίνιο πόσες φορές ανακυκλώνεται χωρίς να χάσει ποιότητα;",
            "options": ["1-2 φορές", "5-7 φορές", "10-15 φορές", "Άπειρες φορές ♾️"],
            "answer": "Άπειρες φορές ♾️",
            "explanation": "Το αλουμίνιο ανακυκλώνεται επ’ άπειρον."
        },
    ]

    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
        st.session_state.user_answers = {}
        st.session_state.score = 0

    for idx, q in enumerate(questions):
        st.markdown(
            f"""
            <div style='background-color:white; padding:15px; border-radius:10px; margin-bottom:15px;'>
            <h4>❓ Ερώτηση {idx+1}: {q['question']}</h4>
            """,
            unsafe_allow_html=True
        )

        st.session_state.user_answers[idx] = st.radio(
            "Επιλέξτε:",
            q["options"],
            key=f"quiz_{idx}"
        )
        st.markdown("</div>", unsafe_allow_html=True)

    if st.button("📤 Υποβολή Απαντήσεων", use_container_width=True):
        st.session_state.quiz_submitted = True
        st.session_state.score = 0

        for idx, q in enumerate(questions):
            if st.session_state.user_answers.get(idx) == q["answer"]:
                st.session_state.score += 1

        if st.session_state.score == len(questions):
            st.balloons()

    if st.session_state.quiz_submitted:
        st.success(f"Σκορ: {st.session_state.score}/{len(questions)}")

# --------------------------
# Section 3: Simple Game Link (Button visible)
# --------------------------
elif section == "Παιχνίδι ♻️":
    st.subheader("🎮 Παιχνίδι Ανακύκλωσης")
    st.write("Πατάς το κουμπί και ανοίγει το παιχνίδι σε νέο tab.")

    game_url = "https://example.com"  # βάλε εδώ το αληθινό link σου

    # Αν υπάρχει link_button στη δική σου έκδοση Streamlit, τέλειο.
    if hasattr(st, "link_button"):
        st.link_button("Παίξε το παιχνίδι ♻️", game_url, use_container_width=True)
    else:
        # fallback (παίζει σε παλιότερες εκδόσεις)
        if st.button("Παίξε το παιχνίδι ♻️", use_container_width=True):
            st.components.v1.html(
                f"<script>window.open('{game_url}', '_blank');</script>",
                height=0
            )

# --------------------------
# Section 4: Eco-Tips
# --------------------------
elif section == "Eco-Tips 💡":
    st.subheader("💡 Συμβουλές για Πράσινη Ζωή")
    st.markdown(
        """
        <div style='background-color:rgba(255,255,255,0.7); padding:15px; border-radius:10px; margin-bottom:20px;'>
        <p>Μικρές αλλαγές, μεγάλη διαφορά!</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    tip_categories = {
        "🏠 Σπίτι": [
            "Χρησιμοποιήστε πλυντήριο μόνο με πλήρες φορτίο",
            "Αποσυνδέστε συσκευές όταν δεν τις χρησιμοποιείτε",
            "Μειώστε το νερό στο ντους"
        ],
        "🛒 Ψώνια": [
            "Υφασμάτινη τσάντα αντί για πλαστική",
            "Προϊόντα με λιγότερη συσκευασία",
            "Τοπικά προϊόντα"
        ],
        "♻️ Ανακύκλωση": [
            "Πλύνετε συσκευασίες πριν τις ανακυκλώσετε",
            "Μάθετε τους κανόνες του δήμου σας",
            "Επαναχρησιμοποιήστε δοχεία"
        ]
    }

    selected_category = st.selectbox("Επιλέξτε κατηγορία:", list(tip_categories.keys()))
    st.markdown(f"<h4 style='color:#2E7D32;'>{selected_category}</h4>", unsafe_allow_html=True)

    for tip in tip_categories[selected_category]:
        st.markdown(
            f"""
            <div style='background-color:white; padding:10px 15px; border-radius:10px; margin-bottom:10px;
                        border-left: 4px solid #4CAF50;'>
            <p style='margin:0;'>🌱 {tip}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

# --------------------------
# Footer
# --------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; padding:20px 0;'>
        <p>🌿 Κάντε τον πλανήτη πιο πράσινο μαζί μας!</p>
        <p style='margin-top:20px; font-size:0.9rem; color:#666;'>© 2025 EcoGuard AI</p>
    </div>
    """,
    unsafe_allow_html=True
)
