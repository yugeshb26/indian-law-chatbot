import streamlit as st
import json
import time
import base64
import os
from datetime import datetime

from db import init_db, create_chat, get_all_chats, get_chat, get_messages
from db import append_message, update_chat_title, delete_chat
from gemini_engine import stream_response, regenerate_response, generate_title, init_rotator

# ── CONFIG ────────────────────────────────────────────────────────────────────
# API keys loaded from Streamlit secrets (never hardcoded in public repos)
API_KEYS = list(st.secrets["GEMINI_API_KEYS"])
rotator = init_rotator(API_KEYS)
API_KEY = API_KEYS[0]
DATASET_PATH = "Alpie-core_core_indian_law.json"
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Indian Law Chatbot",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Init database ────────────────────────────────────────────────────────────
init_db()

# ── Load Ambedkar background ────────────────────────────────────────────────
@st.cache_data
def get_bg_image():
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ambedkar_bg.jpg")
    if os.path.exists(img_path):
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

bg_b64 = get_bg_image()

if bg_b64:
    st.markdown(
        f"""
        <style>
        [data-testid="stMain"] {{
            background:
                linear-gradient(rgba(255, 253, 248, 0.90), rgba(255, 243, 230, 0.90)),
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

# ── Custom CSS — Indian Law Theme ────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar — ChatGPT-style slide panel ─────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B1464 0%, #0C2340 50%, #002147 100%);
    color: white;
    transition: transform 0.3s ease, opacity 0.3s ease !important;
    z-index: 999 !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: white !important;
}

/* ── Hero banner ─────────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #FF9933 0%, #C41E3A 100%);
    padding: 1.8rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(196, 30, 58, 0.25);
}
.hero-banner h1 { color: white; font-size: 1.8rem; font-weight: 700; margin: 0; }
.hero-banner p { color: rgba(255,255,255,0.9); font-size: 0.95rem; margin: 0.4rem 0 0 0; }
.hero-banner::before { content: "☸"; display: block; font-size: 2.5rem; margin-bottom: 0.3rem; opacity: 0.25; }

/* ── Chat bubbles ────────────────────────────────────────────── */
.user-bubble {
    background: linear-gradient(135deg, #FF9933 0%, #E07C24 100%);
    color: white; padding: 0.9rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    margin: 0.5rem 0 0.5rem 20%;
    font-size: 0.95rem; line-height: 1.5;
    box-shadow: 0 2px 12px rgba(255, 153, 51, 0.25);
    word-wrap: break-word;
}
.bot-bubble {
    background: #FFF8F0; color: #1B1464;
    padding: 0.9rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    margin: 0.5rem 20% 0.5rem 0;
    font-size: 0.95rem; line-height: 1.6;
    border: 1px solid #E8D5B7; border-left: 3px solid #C8A84E;
    box-shadow: 0 1px 6px rgba(0,0,0,0.04);
    word-wrap: break-word;
}

/* ── Chat history items in sidebar ───────────────────────────── */
.chat-item {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    margin-bottom: 0.4rem;
    cursor: pointer;
    transition: all 0.2s;
}
.chat-item:hover {
    background: rgba(255, 153, 51, 0.2);
    border-color: #FF9933;
}
.chat-item.active {
    background: rgba(255, 153, 51, 0.25);
    border-color: #FF9933;
    border-left: 3px solid #FF9933;
}
.chat-item .chat-title {
    font-size: 0.85rem; font-weight: 600;
    color: white; margin: 0;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.chat-item .chat-date {
    font-size: 0.7rem; color: rgba(255,255,255,0.5); margin: 0.15rem 0 0 0;
}

/* ── Stat cards ──────────────────────────────────────────────── */
.stat-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(200, 168, 78, 0.3);
    border-radius: 12px; padding: 0.8rem; text-align: center;
}
.stat-card .stat-num { font-size: 1.3rem; font-weight: 700; color: #C8A84E; }
.stat-card .stat-label { font-size: 0.7rem; color: rgba(255,255,255,0.7); text-transform: uppercase; }

/* ── Sidebar buttons ─────────────────────────────────────────── */
section[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    color: white !important;
    border-radius: 10px !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    transition: all 0.2s !important;
}
section[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255, 153, 51, 0.25) !important;
    border-color: #FF9933 !important;
}

/* ── Chat input ──────────────────────────────────────────────── */
.stChatInput > div {
    border-radius: 25px !important;
    border: 2px solid rgba(255, 153, 51, 0.35) !important;
    transition: border 0.3s;
}
.stChatInput > div:focus-within {
    border-color: #FF9933 !important;
    box-shadow: 0 0 0 3px rgba(255, 153, 51, 0.15) !important;
}

/* ── Action buttons (regenerate, continue) ───────────────────── */
.action-row { display: flex; gap: 0.5rem; margin: 0.4rem 0 0.8rem 0; }

/* ── Tricolor bar ────────────────────────────────────────────── */
.tricolor-bar {
    height: 4px;
    background: linear-gradient(90deg, #FF9933 33%, #FFFFFF 33%, #FFFFFF 66%, #138808 66%);
    border-radius: 2px; margin-bottom: 1rem;
}

/* ── Streaming cursor ────────────────────────────────────────── */
@keyframes blink { 50% { opacity: 0; } }
.streaming-cursor::after {
    content: "▊"; animation: blink 0.8s infinite; color: #FF9933;
}

/* ── Thinking pulse animation ────────────────────────────────── */
@keyframes pulse { 0%, 100% { opacity: 0.5; } 50% { opacity: 1; } }
.bot-bubble span[style*="Thinking"] {
    animation: pulse 1.2s ease-in-out infinite;
}

/* ── Sidebar open/close buttons — always visible and styled ──── */
[data-testid="stSidebarCollapsedControl"],
[data-testid="stSidebarCollapseButton"],
button[kind="headerNoPadding"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
}

/* Style the expand button (shown when sidebar is closed) */
[data-testid="stSidebarCollapsedControl"] {
    position: fixed !important;
    top: 10px !important;
    left: 10px !important;
    z-index: 1001 !important;
}
[data-testid="stSidebarCollapsedControl"] button {
    background: linear-gradient(135deg, #1B1464, #0C2340) !important;
    color: white !important;
    border: 2px solid rgba(255, 153, 51, 0.4) !important;
    border-radius: 12px !important;
    width: 44px !important;
    height: 44px !important;
    box-shadow: 0 4px 15px rgba(0,0,0,0.25) !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebarCollapsedControl"] button:hover {
    background: linear-gradient(135deg, #FF9933, #E07C24) !important;
    border-color: #FF9933 !important;
}
[data-testid="stSidebarCollapsedControl"] button svg {
    fill: white !important;
    stroke: white !important;
}

/* Style the collapse button inside sidebar (X to close) */
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    color: white !important;
    border-radius: 10px !important;
    width: 38px !important;
    height: 38px !important;
    transition: all 0.2s !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button:hover {
    background: rgba(255, 153, 51, 0.3) !important;
    border-color: #FF9933 !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] svg {
    fill: white !important;
    stroke: white !important;
}

/* ════════════════════════════════════════════════════════════════
   RESPONSIVE DESIGN — Mobile / Tablet / Desktop
   ════════════════════════════════════════════════════════════════ */

/* ── Mobile (up to 768px) ────────────────────────────────────── */
@media (max-width: 768px) {
    /* Sidebar slides over content on mobile */
    section[data-testid="stSidebar"] {
        min-width: 280px !important;
        max-width: 85vw !important;
        box-shadow: 4px 0 25px rgba(0,0,0,0.3) !important;
    }

    /* Full-width chat bubbles on small screens */
    .user-bubble {
        margin: 0.4rem 0 0.4rem 8%;
        padding: 0.7rem 1rem;
        font-size: 0.9rem;
        border-radius: 16px 16px 4px 16px;
    }
    .bot-bubble {
        margin: 0.4rem 8% 0.4rem 0;
        padding: 0.7rem 1rem;
        font-size: 0.9rem;
        border-radius: 16px 16px 16px 4px;
    }

    /* Compact hero banner */
    .hero-banner {
        padding: 1.2rem 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        margin-top: 3.5rem;
    }
    .hero-banner h1 { font-size: 1.3rem; }
    .hero-banner p { font-size: 0.85rem; }
    .hero-banner::before { font-size: 1.8rem; margin-bottom: 0.2rem; }

    /* Tricolor bar thinner */
    .tricolor-bar { height: 3px; margin-bottom: 0.6rem; }

    /* Stat cards smaller */
    .stat-card { padding: 0.5rem; }
    .stat-card .stat-num { font-size: 1.1rem; }
    .stat-card .stat-label { font-size: 0.6rem; }

    /* Chat input closer to edge */
    .stChatInput > div { border-radius: 20px !important; }

    section[data-testid="stSidebar"] .stButton > button {
        font-size: 0.8rem !important;
        padding: 0.4rem 0.6rem !important;
    }

    /* Reduce main padding */
    .stMainBlockContainer { padding: 0.5rem 0.8rem !important; }

    /* Stack action buttons vertically */
    .action-row { flex-direction: column; }
}

/* ── Small phones (up to 480px) ──────────────────────────────── */
@media (max-width: 480px) {
    .user-bubble {
        margin: 0.3rem 0 0.3rem 4%;
        padding: 0.6rem 0.8rem;
        font-size: 0.85rem;
    }
    .bot-bubble {
        margin: 0.3rem 4% 0.3rem 0;
        padding: 0.6rem 0.8rem;
        font-size: 0.85rem;
    }

    .hero-banner {
        padding: 1rem 0.8rem;
        border-radius: 10px;
    }
    .hero-banner h1 { font-size: 1.1rem; }
    .hero-banner p { font-size: 0.8rem; }
    .hero-banner::before { font-size: 1.5rem; }

    .stat-card .stat-num { font-size: 1rem; }

    /* Ensure text doesn't overflow */
    .stMarkdown, .stChatMessage { word-break: break-word; overflow-wrap: anywhere; }

    .stMainBlockContainer { padding: 0.3rem 0.5rem !important; }
}

/* ── Tablet (769px to 1024px) ────────────────────────────────── */
@media (min-width: 769px) and (max-width: 1024px) {
    .user-bubble { margin: 0.5rem 0 0.5rem 12%; }
    .bot-bubble { margin: 0.5rem 12% 0.5rem 0; }

    .hero-banner { padding: 1.5rem 1.5rem; }
    .hero-banner h1 { font-size: 1.6rem; }

    section[data-testid="stSidebar"] {
        min-width: 260px !important;
        max-width: 300px !important;
    }
}

/* ── Large desktop (1400px+) ─────────────────────────────────── */
@media (min-width: 1400px) {
    .user-bubble { margin: 0.5rem 0 0.5rem 25%; }
    .bot-bubble { margin: 0.5rem 25% 0.5rem 0; }

    .hero-banner { padding: 2rem 2.5rem; }
    .hero-banner h1 { font-size: 2rem; }
}

/* ── Touch-friendly for all mobile ───────────────────────────── */
@media (hover: none) and (pointer: coarse) {
    /* Larger tap targets */
    section[data-testid="stSidebar"] .stButton > button {
        min-height: 44px !important;
        padding: 0.5rem 0.8rem !important;
    }
    .stChatInput > div { min-height: 48px !important; }

    /* No hover effects on touch */
    .chat-item:hover { background: rgba(255,255,255,0.06); }
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.08) !important;
    }
}

/* ── Safe area for notched phones (iPhone etc.) ──────────────── */
@supports (padding: env(safe-area-inset-bottom)) {
    .stChatInput {
        padding-bottom: env(safe-area-inset-bottom) !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ── Load dataset ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_context() -> str:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Load up to 200 entries for rich context (BNS, BNSS, BSA, Constitution)
    return "\n".join(
        f"Q: {item['prompt']}\nA: {item['response']}"
        for item in data[:200]
    )

try:
    dataset_context = load_context()
except FileNotFoundError:
    st.error(f"File not found: `{DATASET_PATH}`")
    st.stop()

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
    "Always provide complete, thorough responses. Do not cut off mid-sentence.\n\n"
    "Dataset Context:\n" + dataset_context
)

# ── Session state init ───────────────────────────────────────────────────────
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def load_chat(chat_id: str):
    """Load a chat from DB into session state."""
    st.session_state.active_chat_id = chat_id
    db_msgs = get_messages(chat_id)
    st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in db_msgs]


def start_new_chat():
    """Create a fresh chat."""
    st.session_state.active_chat_id = None
    st.session_state.messages = []


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚖️ Indian Law Bot")
    st.markdown(
        "<p style='color:rgba(255,255,255,0.7); font-size:0.85rem;'>"
        "Your AI-powered legal assistant</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # New Chat button
    if st.button("✨  New Chat", key="new_chat", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.markdown("---")

    # Stats
    all_chats = get_all_chats()
    total_chats = len(all_chats)
    total_msgs = sum(1 for m in st.session_state.messages if m["role"] == "user")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{total_chats}</div>'
            f'<div class="stat-label">Chats</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="stat-card"><div class="stat-num">{total_msgs}</div>'
            f'<div class="stat-label">Questions</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Chat History List
    st.markdown("### Chat History")
    if not all_chats:
        st.markdown(
            "<p style='color:rgba(255,255,255,0.4); font-size:0.8rem;'>No chats yet. Start asking!</p>",
            unsafe_allow_html=True,
        )
    else:
        for chat in all_chats:
            is_active = chat["chat_id"] == st.session_state.active_chat_id
            active_cls = " active" if is_active else ""
            try:
                dt = datetime.fromisoformat(chat["updated_at"])
                date_str = dt.strftime("%b %d, %I:%M %p")
            except Exception:
                date_str = chat["updated_at"][:16]

            # Each chat is a button
            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                if st.button(
                    f"💬 {chat['title'][:30]}",
                    key=f"chat_{chat['chat_id']}",
                    use_container_width=True,
                ):
                    load_chat(chat["chat_id"])
                    st.rerun()
            with col_del:
                if st.button("🗑", key=f"del_{chat['chat_id']}"):
                    delete_chat(chat["chat_id"])
                    if st.session_state.active_chat_id == chat["chat_id"]:
                        start_new_chat()
                    st.rerun()

    st.markdown("---")

    # Quick Topics
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
    st.markdown(
        "<p style='color:rgba(255,255,255,0.5); font-size:0.75rem; text-align:center;'>"
        "Powered by Gemini 2.5 Flash<br>Not a substitute for legal advice</p>",
        unsafe_allow_html=True,
    )

# ── Hero Banner (only when no messages) ──────────────────────────────────────
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

# ── Render chat history ─────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

# ── Regenerate & Continue buttons (after last assistant message) ─────────────
if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant":
    col_regen, col_cont, _ = st.columns([1, 1, 3])
    with col_regen:
        regen_clicked = st.button("🔄 Regenerate", key="regen_btn")
    with col_cont:
        cont_clicked = st.button("➡️ Continue", key="cont_btn")
else:
    regen_clicked = False
    cont_clicked = False

# ── Handle quick topic clicks ────────────────────────────────────────────────
pending = st.session_state.pop("pending_topic", None)

# ── Chat input ───────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask anything about Indian law...")
active_prompt = prompt or pending


# ── Process: New message / Regenerate / Continue ─────────────────────────────

def stream_and_display(messages_for_api: list[dict]) -> str:
    """Stream Gemini response with live display, return full text."""
    # Show loading indicator immediately
    loading = st.empty()
    loading.markdown(
        '<div class="bot-bubble" style="opacity:0.7;">'
        '<span style="color:#C8A84E;">⚖️ Thinking...</span></div>',
        unsafe_allow_html=True,
    )

    placeholder = st.empty()
    full_response = ""
    start = time.time()

    try:
        first_chunk = True
        for chunk in stream_response(API_KEY, SYSTEM_PROMPT, messages_for_api):
            if first_chunk:
                loading.empty()  # Remove "Thinking..." once streaming starts
                first_chunk = False
            full_response += chunk
            placeholder.markdown(
                f'<div class="bot-bubble streaming-cursor">{full_response}</div>',
                unsafe_allow_html=True,
            )

        elapsed = time.time() - start
        placeholder.markdown(
            f'<div class="bot-bubble">{full_response}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Response time: {elapsed:.1f}s")
        return full_response

    except Exception as e:
        loading.empty()
        elapsed = time.time() - start
        if full_response:
            placeholder.markdown(
                f'<div class="bot-bubble">{full_response}</div>',
                unsafe_allow_html=True,
            )
            st.warning(f"Response may be incomplete ({elapsed:.1f}s). Click 'Continue' to extend.")
            return full_response
        else:
            placeholder.empty()
            st.error(f"Failed to get response: {str(e)[:200]}")
            return ""


if active_prompt:
    # ── New user message ─────────────────────────────────────────────────────
    # Create chat in DB if this is a new conversation
    if not st.session_state.active_chat_id:
        chat_id = create_chat("New Chat")
        st.session_state.active_chat_id = chat_id
    else:
        chat_id = st.session_state.active_chat_id

    # Save user message
    st.session_state.messages.append({"role": "user", "content": active_prompt})
    append_message(chat_id, "user", active_prompt)
    st.markdown(f'<div class="user-bubble">{active_prompt}</div>', unsafe_allow_html=True)

    # Build context-aware messages for API (full history)
    api_messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\nUser: " + st.session_state.messages[0]["content"]}]
    for m in st.session_state.messages[1:]:
        api_messages.append(m)

    # Stream response
    reply = stream_and_display(api_messages)

    if reply:
        st.session_state.messages.append({"role": "assistant", "content": reply})
        append_message(chat_id, "assistant", reply)

        # Auto-generate title from first question
        if len(get_messages(chat_id)) <= 2:
            title = generate_title(API_KEY, active_prompt)
            update_chat_title(chat_id, title)

elif regen_clicked and st.session_state.messages:
    # ── Regenerate last response ─────────────────────────────────────────────
    chat_id = st.session_state.active_chat_id
    # Remove last assistant message
    st.session_state.messages.pop()

    api_messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\nUser: " + st.session_state.messages[0]["content"]}]
    for m in st.session_state.messages[1:]:
        api_messages.append(m)

    reply = stream_and_display(api_messages)
    if reply:
        st.session_state.messages.append({"role": "assistant", "content": reply})
        if chat_id:
            append_message(chat_id, "assistant", reply)

elif cont_clicked and st.session_state.messages:
    # ── Continue last response ───────────────────────────────────────────────
    chat_id = st.session_state.active_chat_id
    # Add a continue request
    cont_msg = {"role": "user", "content": "Continue your response. Do not repeat what you already said."}
    st.session_state.messages.append(cont_msg)
    if chat_id:
        append_message(chat_id, "user", cont_msg["content"])

    api_messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\nUser: " + st.session_state.messages[0]["content"]}]
    for m in st.session_state.messages[1:]:
        api_messages.append(m)

    reply = stream_and_display(api_messages)
    if reply:
        st.session_state.messages.append({"role": "assistant", "content": reply})
        if chat_id:
            append_message(chat_id, "assistant", reply)
