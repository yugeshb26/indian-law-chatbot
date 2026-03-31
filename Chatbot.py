import streamlit as st
import json
import time
import base64
import os
from google import genai

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY = st.secrets["GEMINI_API_KEY"]
DATASET_PATH = "Alpie-core_core_indian_law.json"
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Indian Law Chatbot",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Load Ambedkar background image as base64 ─────────────────────────────────
@st.cache_data
def get_bg_image():
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ambedkar_bg.jpg")
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

bg_b64 = get_bg_image()

# ── Ambedkar faded background (injected as a fixed div behind content) ───────
if bg_b64:
    st.markdown(
        f"""
        <style>
        /* ── Ambedkar portrait blended as watermark ────────────── */
        [data-testid="stMain"] {{
            background:
                linear-gradient(
                    rgba(255, 253, 248, 0.90),
                    rgba(255, 243, 230, 0.90)
                ),
                url("data:image/jpeg;base64,{bg_b64}");
            background-size: cover, 380px auto;
            background-position: center, center center;
            background-repeat: no-repeat, no-repeat;
            background-attachment: fixed, fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ── Custom CSS for clean, colorful UI ────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit extras ──────────────────────────────── */
#MainMenu, footer, header {visibility: hidden;}

/* ── Sidebar — deep navy (Ashoka Chakra / authority) ───────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B1464 0%, #0C2340 50%, #002147 100%);
    color: white;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: white !important;
}

/* ── Hero banner — saffron-to-maroon (tricolor + legal robes) ──── */
.hero-banner {
    background: linear-gradient(135deg, #FF9933 0%, #C41E3A 100%);
    padding: 1.8rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(196, 30, 58, 0.25);
}
.hero-banner h1 {
    color: white;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero-banner p {
    color: rgba(255,255,255,0.9);
    font-size: 0.95rem;
    margin: 0.4rem 0 0 0;
}

/* ── Ashoka Chakra decorative ring ─────────────────────────────── */
.hero-banner::before {
    content: "☸";
    display: block;
    font-size: 2.5rem;
    margin-bottom: 0.3rem;
    opacity: 0.25;
}

/* ── Chat bubbles — saffron user, cream bot ────────────────────── */
.user-bubble {
    background: linear-gradient(135deg, #FF9933 0%, #E07C24 100%);
    color: white;
    padding: 0.9rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0 0.5rem 20%;
    font-size: 0.95rem;
    line-height: 1.5;
    box-shadow: 0 2px 12px rgba(255, 153, 51, 0.25);
    word-wrap: break-word;
}
.bot-bubble {
    background: #FFF8F0;
    color: #1B1464;
    padding: 0.9rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.5rem 20% 0.5rem 0;
    font-size: 0.95rem;
    line-height: 1.6;
    border: 1px solid #E8D5B7;
    border-left: 3px solid #C8A84E;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    word-wrap: break-word;
}

/* ── Stat cards — gold accents ─────────────────────────────────── */
.stat-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(200, 168, 78, 0.3);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    margin-bottom: 0.5rem;
}
.stat-card .stat-num {
    font-size: 1.5rem;
    font-weight: 700;
    color: #C8A84E;
}
.stat-card .stat-label {
    font-size: 0.75rem;
    color: rgba(255,255,255,0.7);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Topic buttons — saffron hover ─────────────────────────────── */
section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    color: white !important;
    border-radius: 10px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    transition: all 0.2s !important;
}
section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
    background: rgba(255, 153, 51, 0.25) !important;
    border-color: #FF9933 !important;
}

/* ── Chat input — saffron accent ───────────────────────────────── */
.stChatInput > div {
    border-radius: 25px !important;
    border: 2px solid rgba(255, 153, 51, 0.35) !important;
    transition: border 0.3s;
}
.stChatInput > div:focus-within {
    border-color: #FF9933 !important;
    box-shadow: 0 0 0 3px rgba(255, 153, 51, 0.15) !important;
}

/* ── Spinner — saffron ─────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: #FF9933 !important;
}

/* ── Sidebar clear button — deep maroon ────────────────────────── */
section[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #8B0000, #6B0F1A) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    width: 100%;
    transition: transform 0.2s;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 4px 15px rgba(139, 0, 0, 0.35);
}

/* ── Tricolor accent bar at top ────────────────────────────────── */
.tricolor-bar {
    height: 4px;
    background: linear-gradient(90deg, #FF9933 33%, #FFFFFF 33%, #FFFFFF 66%, #138808 66%);
    border-radius: 2px;
    margin-bottom: 1rem;
}
</style>
""", unsafe_allow_html=True)

# ── Load dataset ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_context() -> str:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return "\n".join(
        f"Q: {item['prompt']}\nA: {item['response']}"
        for item in data[:15]
    )

try:
    dataset_context = load_context()
except FileNotFoundError:
    st.error(f"File not found: `{DATASET_PATH}`")
    st.stop()

# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert Indian Law Chatbot assistant with comprehensive knowledge of "
    "Indian legal provisions, acts, and regulations.\n\n"
    "Your responsibilities:\n"
    "1. Answer questions about Indian laws accurately\n"
    "2. Explain complex legal concepts in simple terms\n"
    "3. Reference specific Acts, Sections, and Articles when relevant\n"
    "4. Provide context on how laws are applied in India\n"
    "5. Mention important case law and precedents when applicable\n"
    "6. Clarify constitutional provisions and fundamental rights\n"
    "7. Suggest consulting qualified legal professionals for specific advice\n\n"
    "Dataset Context:\n" + dataset_context
)

# ── Gemini API call ──────────────────────────────────────────────────────────
def ask(question: str) -> str:
    if not API_KEY or len(API_KEY) < 20:
        raise ValueError("API Key not configured")

    client = genai.Client(api_key=API_KEY)
    full_prompt = SYSTEM_PROMPT + f"\n\n--- USER QUESTION ---\n{question}\n\n--- ANSWER ---"

    resp = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[{"role": "user", "parts": [{"text": full_prompt}]}],
        config={"temperature": 0.5, "top_p": 0.85, "max_output_tokens": 1000},
    )

    if not resp or not resp.text:
        raise ValueError("Gemini returned an empty response")
    return resp.text

# ── Session state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Indian Law Bot")
    st.markdown(
        "<p style='color:rgba(255,255,255,0.7); font-size:0.85rem;'>"
        "Your AI-powered legal assistant for Indian law</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # Stats
    total_msgs = len(st.session_state.messages)
    user_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{user_msgs}</div>'
            f'<div class="stat-label">Questions</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{total_msgs - user_msgs}</div>'
            f'<div class="stat-label">Answers</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("### Quick Topics")
    topics = [
        ("⚖️", "Fundamental Rights", "Explain the Fundamental Rights under the Indian Constitution"),
        ("📜", "IPC Sections", "What are the most important IPC sections everyone should know?"),
        ("🔍", "RTI Act", "How do I file an RTI application in India?"),
        ("🏠", "Property Law", "What are the key property laws in India for buying and selling?"),
        ("🛡️", "Consumer Rights", "What are my consumer rights under the Consumer Protection Act?"),
        ("👷", "Labour Law", "What are the major labour laws protecting workers in India?"),
    ]
    for icon, label, question in topics:
        if st.button(f"{icon}  {label}", key=f"topic_{label}", use_container_width=True):
            st.session_state.pending_topic = question
            st.rerun()

    st.markdown("---")
    if st.button("🗑️  Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='color:rgba(255,255,255,0.5); font-size:0.75rem; text-align:center;'>"
        "Powered by Gemini 2.5 Flash<br>Not a substitute for legal advice</p>",
        unsafe_allow_html=True,
    )

# ── Hero Banner ──────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        """
        <div class="tricolor-bar"></div>
        <div class="hero-banner">
            <h1>⚖️ Indian Law Assistant</h1>
            <p>Ask me anything about Indian laws, acts, constitutional provisions, or legal rights</p>
        </div>
        <div class="tricolor-bar"></div>
        """,
        unsafe_allow_html=True,
    )

# ── Render chat history with custom bubbles ──────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

# ── Handle quick topic clicks ────────────────────────────────────────────────
pending = st.session_state.pop("pending_topic", None)

# ── Chat input ───────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask anything about Indian law...")

# Use pending topic if no manual input
active_prompt = prompt or pending

if active_prompt:
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    st.markdown(f'<div class="user-bubble">{active_prompt}</div>', unsafe_allow_html=True)

    with st.spinner("Thinking..."):
        try:
            start = time.time()
            reply = ask(active_prompt)
            elapsed = time.time() - start

            st.markdown(f'<div class="bot-bubble">{reply}</div>', unsafe_allow_html=True)
            st.caption(f"Response time: {elapsed:.1f}s")
            st.session_state.messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            st.error(f"Something went wrong: {str(e)[:200]}")
