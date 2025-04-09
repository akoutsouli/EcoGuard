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
import random
import time

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
# Header & Sidebar with Cool Effects
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

# Sidebar with floating logo effect
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
        ("Ανίχνευση Απόβλητων 🗑️", "Κουίζ Ανακύκλωσης 📝", "Eco-Tips 💡"),
        index=0
    )

# --------------------------
# Waste Categories (Enhanced)
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
# Load CLIP model with Progress
# --------------------------
@st.cache_resource
def load_model():
    with st.spinner("Φόρτωση μοντέλου AI... ⏳"):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        st.success("Το μοντέλο φορτώθηκε με επιτυχία! ✅")
        return model, processor

model, processor = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --------------------------
# Section 1: Waste Detection (Enhanced)
# --------------------------
if section == "Ανίχνευση Απόβλητων 🗑️":
    st.subheader("🔍 Ανίχνευση Απόβλητων με AI")
    st.markdown("""
        <div style='background-color:rgba(255,255,255,0.7); padding:15px; border-radius:10px;'>
        <p>📸 Ανεβάστε μια φωτογραφία και το AI θα σας πει αν είναι ανακυκλώσιμο!</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader(
            "Επιλέξτε εικόνα", 
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
    with col2:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("📸 Τραβήξτε Φωτογραφία", disabled=True):
            st.info("Η λειτουργία αυτή θα είναι διαθέσιμη σύντομα!")
    
    if uploaded_file:
        with st.spinner("Ανάλυση εικόνας... 🔍"):
            image = Image.open(uploaded_file)
            st.image(image, caption="Η εικόνα σας", use_container_width=True)
            
            # Combine labels with emojis
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
            
            # Get top 3 predictions
            top_probs, top_indices = torch.topk(probs, 3)
            
            st.success("🔎 Αποτελέσματα Ανάλυσης")
            
            # Display results in a nice way
            result_col1, result_col2 = st.columns(2)
            
            best_idx = top_indices[0][0].item()
            best_label = candidate_labels[best_idx]
            
            with result_col1:
                if best_label in recyclable_set:
                    st.markdown(f"""
                        <div style='background-color:#E8F5E9; padding:15px; border-radius:10px;'>
                        <h3 style='color:#2E7D32; text-align:center;'>♻️ Ανακυκλώσιμο</h3>
                        <p style='text-align:center; font-size:1.3rem;'><b>{recyclable_set[best_label]}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div style='background-color:#FFEBEE; padding:15px; border-radius:10px;'>
                        <h3 style='color:#C62828; text-align:center;'>🚫 Μη Ανακυκλώσιμο</h3>
                        <p style='text-align:center; font-size:1.3rem;'><b>{non_recyclable_set[best_label]}</b></p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with result_col2:
                st.markdown("""
                    <div style='background-color:white; padding:15px; border-radius:10px;'>
                    <h4 style='color:#2E7D32;'>🔍 Άλλες πιθανότητες:</h4>
                    """, unsafe_allow_html=True)
                
                for i in range(1, 3):
                    idx = top_indices[0][i].item()
                    label = candidate_labels[idx]
                    prob = top_probs[0][i].item()
                    
                    if label in recyclable_set:
                        disp_name = recyclable_set[label]
                    else:
                        disp_name = non_recyclable_set[label]
                    
                    st.markdown(f"""
                        <p>- {disp_name} <span style='float:right;'>{prob:.1%}</span></p>
                        """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Recycling instructions
            st.markdown("---")
            if best_label in recyclable_set:
                st.markdown(f"""
                    <div style='background-color:#E8F5E9; padding:15px; border-radius:10px;'>
                    <h4 style='color:#2E7D32;'>✅ Πώς να ανακυκλώσετε:</h4>
                    <ul>
                        <li>Καθαρίστε το αντικείμενο από υπολείμματα</li>
                        <li>Αφαιρέστε πώματα ή καπάκια</li>
                        <li>Τοποθετήστε στον κατάλληλο κάδο ανακύκλωσης</li>
                    </ul>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div style='background-color:#FFF3E0; padding:15px; border-radius:10px;'>
                    <h4 style='color:#E65100;'>⚠️ Σημαντική Σημείωση:</h4>
                    <p>Αυτό το αντικείμενο δεν πρέπει να ανακυκλωθεί. Αν είναι επικίνδυνο (π.χ. μπαταρίες), 
                    ελέγξτε για ειδικούς σημεία συλλογής στην περιοχή σας.</p>
                    </div>
                    """, unsafe_allow_html=True)

# --------------------------
# Section 2: Recycling Quiz (Enhanced)
# --------------------------
elif section == "Κουίζ Ανακύκλωσης 📝":
    st.subheader("📚 Κουίζ Ανακύκλωσης")
    st.markdown("""
        <div style='background-color:rgba(255,255,255,0.7); padding:15px; border-radius:10px; margin-bottom:20px;'>
        <p>Δοκιμάστε τις γνώσεις σας με αυτό το διαδραστικό κουίζ! Για κάθε σωστή απάντηση, κερδίζετε 10 πόντους.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced quiz questions with explanations
    questions = [
        {
            "question": "Ποιο από τα παρακάτω είναι ανακυκλώσιμο;", 
            "options": [
                "Χαρτί 📄", 
                "Φαγητά 🍲", 
                "Μπαταρίες 🔋", 
                "Οργανικά απόβλητα 🥕"
            ], 
            "answer": "Χαρτί 📄",
            "explanation": "Το χαρτί μπορεί να ανακυκλωθεί έως 7 φορές! Τα υπόλοιπα ανήκουν σε άλλες κατηγορίες."
        },
        {
            "question": "Πού πρέπει να πετάμε τις πλαστικές φιάλες;", 
            "options": [
                "Κάδος Ανακύκλωσης ♻️", 
                "Κάδος Οργανικών 🥕"
            ], 
            "answer": "Κάδος Ανακύκλωσης ♻️",
            "explanation": "Οι πλαστικές φιάλες (με τον αριθμό 1 ή 2 στο πλαστικό) πάνε στον κάδο ανακύκλωσης."
        },
        {
            "question": "Ποιο από τα παρακάτω υλικά χρειάζεται ειδική διαχείριση;", 
            "options": [
                "Πλαστικό 🥤", 
                "Γυαλί 🍷", 
                "Μπαταρίες 🔋", 
                "Χαρτόνι 📦"
            ], 
            "answer": "Μπαταρίες 🔋",
            "explanation": "Οι μπαταρίες περιέχουν επικίνδυνες ουσίες και πρέπει να ανακυκλώνονται σε ειδικά σημεία."
        },
        {
            "question": "Πόσες φορές μπορεί να ανακυκλωθεί το αλουμίνιο χωρίς να χάσει την ποιότητά του;", 
            "options": [
                "1-2 φορές",
                "5-7 φορές",
                "10-15 φορές", 
                "Άπειρες φορές ♾️"
            ], 
            "answer": "Άπειρες φορές ♾️",
            "explanation": "Το αλουμίνιο διατηρεί τις ιδιότητές του επ' αόριστον κατά την ανακύκλωση!"
        },
        {
            "question": "Γιατί είναι σημαντική η σωστή ταξινόμηση των απορριμμάτων;", 
            "options": [
                "Μειώνει τη ρύπανση", 
                "Βελτιώνει την ανακύκλωση", 
                "Ενισχύει τη βιωσιμότητα", 
                "Όλα τα παραπάνω"
            ], 
            "answer": "Όλα τα παραπάνω",
            "explanation": "Η σωστή ανακύκλωση βοηθάει σε όλους αυτούς τους τομείς ταυτόχρονα!"
        }
    ]
    
    # Initialize session state for quiz if not exists
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
        st.session_state.user_answers = {}
        st.session_state.score = 0
    
    # Display quiz questions
    for idx, q in enumerate(questions):
        st.markdown(f"""
            <div style='background-color:white; padding:15px; border-radius:10px; margin-bottom:15px;'>
            <h4>❓ Ερώτηση {idx+1}: {q['question']}</h4>
            """, unsafe_allow_html=True)
        
        st.session_state.user_answers[idx] = st.radio(
            "Επιλέξτε:",
            q["options"],
            key=f"quiz_{idx}",
            index=None if idx not in st.session_state.user_answers else q["options"].index(st.session_state.user_answers[idx])
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Submit button with cool effect
    if st.button("📤 Υποβολή Απαντήσεων", use_container_width=True):
        st.session_state.quiz_submitted = True
        st.session_state.score = 0
        
        for idx, q in enumerate(questions):
            if st.session_state.user_answers[idx] == q["answer"]:
                st.session_state.score += 1
        
        # Show confetti on good score
        if st.session_state.score >= len(questions) * 0.8:
            st.balloons()
        elif st.session_state.score >= len(questions) * 0.5:
            st.snow()
    
    # Show results if submitted
    if st.session_state.quiz_submitted:
        st.markdown(f"""
            <div style='background-color:#E3F2FD; padding:20px; border-radius:10px; text-align:center;'>
            <h2 style='color:#0D47A1;'>🏆 Σκορ: {st.session_state.score}/{len(questions)}</h2>
            <p style='font-size:1.2rem;'>{'🎉 Τέλειο! Είστε ειδικός στην ανακύκλωση!' if st.session_state.score == len(questions) else '👍 Καλή δουλειά!' if st.session_state.score >= len(questions)*0.7 else '💪 Μπορείτε να τα πάτε καλύτερα!'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📖 Απαντήσεις & Επεξηγήσεις")
        
        for idx, q in enumerate(questions):
            with st.expander(f"Ερώτηση {idx+1}: {q['question']}"):
                user_answer = st.session_state.user_answers[idx]
                is_correct = user_answer == q["answer"]
                
                st.markdown(f"""
                    <div style='background-color:{"#E8F5E9" if is_correct else "#FFEBEE"}; 
                                padding:15px; border-radius:10px; margin-bottom:10px;'>
                    <p><b>Η απάντησή σας:</b> {user_answer} {"✅" if is_correct else "❌"}</p>
                    <p><b>Σωστή απάντηση:</b> {q['answer']}</p>
                    </div>
                    <p>{q['explanation']}</p>
                    """, unsafe_allow_html=True)

# --------------------------
# New Section: Eco-Tips
# --------------------------
elif section == "Eco-Tips 💡":
    st.subheader("💡 Συμβουλές για Πράσινη Ζωή")
    st.markdown("""
        <div style='background-color:rgba(255,255,255,0.7); padding:15px; border-radius:10px; margin-bottom:20px;'>
        <p>Μικρές αλλαγές, μεγάλη διαφορά! Δείτε πώς μπορείτε να βοηθήσετε τον πλανήτη.</p>
        </div>
        """, unsafe_allow_html=True)
    
    tip_categories = {
        "🏠 Σπίτι": [
            "Χρησιμοποιήστε πλυντήρια και πλυντήριο πιάτων μόνο με πλήρες φορτίο",
            "Αποσυνδέστε ηλεκτρονικές συσκευές όταν δεν τις χρησιμοποιείτε",
            "Εγκαταστήστε εξοικονόμηση νερού στις βρύσες"
        ],
        "🛒 Ψώνια": [
            "Χρησιμοποιήστε υφασμάτινα τσάντα αντί για πλαστικά",
            "Επιλέξτε προϊόντα με λιγότερη συσκευασία",
            "Αγοράζετε τοπικά για να μειώσετε τον αποτύπωμα άνθρακα"
        ],
        "♻️ Ανακύκλωση": [
            "Πλύνετε συσκευασίες πριν τις ανακυκλώσετε",
            "Μάθετε τους κανόνες ανακύκλωσης της περιοχής σας",
            "Επαναχρησιμοποιήστε δοχεία για αποθήκευση"
        ]
    }
    
    selected_category = st.selectbox(
        "Επιλέξτε κατηγορία συμβουλών:",
        list(tip_categories.keys())
    )
    
    st.markdown(f"<h4 style='color:#2E7D32;'>{selected_category} Tips</h4>", unsafe_allow_html=True)
    
    for tip in tip_categories[selected_category]:
        st.markdown(f"""
            <div style='background-color:white; padding:10px 15px; border-radius:10px; margin-bottom:10px; 
                        border-left: 4px solid #4CAF50;'>
            <p style='margin:0;'>🌱 {tip}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.subheader("📊 Οδηγός Ανακύκλωσης")
    
    recycling_data = {
        "Υλικό": ["Χαρτί", "Πλαστικό", "Γυαλί", "Μέταλλο", "Ηλεκτρονικά"],
        "Σύμβολο": ["📄", "🥤", "🍷", "🥫", "💻"],
        "Πού να το πετάξετε": ["Μπλε κάδος", "Κίτρινος κάδος", "Πράσινος κάδος", "Ειδικοί κάδοι", "Ειδικά σημεία"],
        "Σημειώσεις": [
            "Χωρίς λιπαντικά ή τρόφιμα",
            "Μόνο με αριθμούς 1-7",
            "Χωρίς κρύσταλλα ή τζάμια",
            "Συμπιέστε κουτιά",
            "Ποτέ στον γενικό κάδο"
        ]
    }
    
    st.table(recycling_data)

# --------------------------
# Footer with Social Links
# --------------------------
st.markdown("---")
st.markdown(
    """
    <div style='text-align:center; padding:20px 0;'>
        <p>🌿 Κάντε τον πλανήτη πιο πράσινο μαζί μας!</p>
        <div style='display:flex; justify-content:center; gap:15px; margin-top:10px;'>
            <a href='https://www.instagram.com/ecoguardai/'>📱 Instagram</a>
            <a href='https://www.facebook.com/ecoguardai'>📘 Facebook</a>
            <a href='https://x.com/ecoguard_ai'>🐦 Twitter</a>
            <a href='https://www.tiktok.com/@ecoguardai'>⌚ TikTok</a>
        </div>
        <p style='margin-top:20px; font-size:0.9rem; color:#666;'>© 2025 EcoGuard AI - Όλα τα δικαιώματα διατηρούνται</p>
    </div>
    """,
    unsafe_allow_html=True
)
