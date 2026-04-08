import streamlit as st
import json
import time
import base64
import os
import urllib.parse
import markdown as md_lib
from datetime import datetime

from db import init_db, create_chat, get_all_chats, get_chat, get_messages
from db import append_message, update_chat_title, delete_chat
from gemini_engine import stream_response, regenerate_response, generate_title, init_rotator

# ── CONFIG ────────────────────────────────────────────────────────────────────
# API keys: try environment variable first (Render), then Streamlit secrets
if os.environ.get("GEMINI_API_KEYS"):
    import json as _json
    API_KEYS = _json.loads(os.environ["GEMINI_API_KEYS"])
else:
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
    # Aether Neon: dark immersive background with subtle Ambedkar overlay + grid
    st.markdown(
        f"""
        <style>
        [data-testid="stMain"] {{
            background:
                radial-gradient(ellipse at top left, rgba(180, 90, 237, 0.18) 0%, transparent 45%),
                radial-gradient(ellipse at bottom right, rgba(0, 200, 255, 0.18) 0%, transparent 45%),
                radial-gradient(ellipse at center, rgba(255, 184, 77, 0.10) 0%, transparent 60%),
                linear-gradient(180deg, rgba(7, 9, 18, 0.92) 0%, rgba(10, 14, 39, 0.94) 100%),
                url("data:image/jpeg;base64,{bg_b64}");
            background-size: cover, cover, cover, cover, 480px auto;
            background-position: center, center, center, center, center 30%;
            background-repeat: no-repeat, no-repeat, no-repeat, no-repeat, no-repeat;
            background-attachment: fixed, fixed, fixed, fixed, fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# ── Custom CSS — Aether Neon (loaded from styles.css) ───────────────────────
@st.cache_data
def load_css() -> str:
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "styles.css")
    with open(css_path, "r", encoding="utf-8") as f:
        return f.read()

st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)

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

# ── Markdown + bubble rendering helpers ─────────────────────────────────────
import html as _html


def md_to_html(text: str, safe: bool = True) -> str:
    """Convert markdown text to HTML for embedding in chat bubbles.

    safe=True escapes raw HTML first (use for user input).
    safe=False trusts the input (use for LLM output that we want fully formatted).
    """
    if safe:
        text = _html.escape(text)
    return md_lib.markdown(
        text,
        extensions=["fenced_code", "tables", "nl2br", "sane_lists"],
    )


def _copy_button(content: str) -> str:
    """Render a small 'Copy' button that copies the original plain text."""
    encoded = urllib.parse.quote(content)
    return (
        '<button class="copy-btn" '
        f'onclick="navigator.clipboard.writeText(decodeURIComponent(\'{encoded}\'));'
        "this.innerText='✓ Copied';"
        "setTimeout(()=>this.innerText='📋 Copy',1800);\">"
        "📋 Copy</button>"
    )


def user_bubble_html(content: str) -> str:
    body = md_to_html(content, safe=True)
    return (
        '<div class="msg-row user-row">'
        f'<div class="user-bubble">{body}</div>'
        '<div class="avatar user-avatar">👤</div>'
        '</div>'
    )


def bot_bubble_html(content: str, streaming: bool = False) -> str:
    # Escape HTML first to prevent any raw <script> from LLM output (defense-in-depth)
    body = md_to_html(content, safe=True) if content else ""
    streaming_cls = " streaming-cursor" if streaming else ""
    copy_html = _copy_button(content) if (content and not streaming) else ""
    return (
        '<div class="msg-row bot-row">'
        '<div class="avatar bot-avatar">⚖️</div>'
        f'<div class="bot-bubble{streaming_cls}">{body}{copy_html}</div>'
        '</div>'
    )


# ── Session state init ───────────────────────────────────────────────────────
# Defaults for all session keys we use (avoids KeyError edge cases)
for _key, _default in [
    ("active_chat_id", None),
    ("messages", []),
    ("pending_response", False),  # True while we still need to stream a reply
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ── Restore active chat from URL query params (survives reconnects) ─────────
# On Render free tier the app sleeps; when it wakes the WebSocket reconnects
# with a fresh session, dropping in-memory state. Persisting the active chat
# ID in the URL means we can recover it after a reload/reconnect so the user
# does NOT get a brand-new chat row for every question.
_qp_chat = st.query_params.get("chat")
if _qp_chat and st.session_state.active_chat_id is None:
    _existing = get_chat(_qp_chat)
    if _existing:
        st.session_state.active_chat_id = _qp_chat
        _db_msgs = get_messages(_qp_chat)
        st.session_state.messages = [
            {"role": m["role"], "content": m["content"]} for m in _db_msgs
        ]


def _sync_url():
    """Reflect the active chat in the URL so it survives reconnects."""
    if st.session_state.active_chat_id:
        st.query_params["chat"] = st.session_state.active_chat_id
    else:
        st.query_params.clear()


def load_chat(chat_id: str):
    """Load a chat from DB into session state."""
    st.session_state.active_chat_id = chat_id
    db_msgs = get_messages(chat_id)
    st.session_state.messages = [{"role": m["role"], "content": m["content"]} for m in db_msgs]
    st.session_state.pending_response = False
    _sync_url()


def start_new_chat():
    """Create a fresh chat."""
    st.session_state.active_chat_id = None
    st.session_state.messages = []
    st.session_state.pending_response = False
    _sync_url()


def submit_user_input(text: str):
    """Save a new user message and queue a response. Caller should rerun."""
    if not st.session_state.active_chat_id:
        st.session_state.active_chat_id = create_chat("New Chat")
        _sync_url()
    st.session_state.messages.append({"role": "user", "content": text})
    append_message(st.session_state.active_chat_id, "user", text)
    st.session_state.pending_response = True


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
            submit_user_input(question)
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='color:var(--text-dim); font-size:0.72rem; text-align:center; "
        "letter-spacing:0.1em; text-transform:uppercase; font-weight:600;'>"
        "⚡ Powered by Gemini<br>"
        "<span style='font-size:0.65rem; opacity:0.7; text-transform:none; letter-spacing:normal;'>"
        "Aether Neon · Legal AI</span></p>",
        unsafe_allow_html=True,
    )

# ── Disclaimer banner (always visible at top) ───────────────────────────────
st.markdown(
    '<div class="disclaimer-banner">'
    '<strong>⚠️ Legal Disclaimer:</strong> This AI assistant provides general legal information '
    'about Indian law for educational purposes only. It is <strong>not a substitute</strong> for '
    'advice from a qualified advocate. Always consult a licensed legal professional for specific cases.'
    '</div>',
    unsafe_allow_html=True,
)

# ── Hero Banner + Welcome Cards (only when no messages) ─────────────────────
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

    st.markdown('<p class="welcome-label">💡 Try asking about</p>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-state">', unsafe_allow_html=True)

    welcome_examples = [
        ("⚖️", "Fundamental Rights",
         "What are the Fundamental Rights guaranteed by the Indian Constitution?"),
        ("📜", "Article 21",
         "Explain Article 21 of the Indian Constitution and the right to life and liberty."),
        ("🚨", "How to file an FIR",
         "What is the step-by-step procedure to file an FIR in India?"),
        ("💍", "Divorce process",
         "What is the legal process for filing for divorce in India?"),
        ("🏠", "Property inheritance",
         "What are the property inheritance rights for daughters under Hindu law?"),
        ("💼", "Workplace harassment",
         "What legal protections exist against sexual harassment at the workplace in India?"),
    ]

    cols = st.columns(3)
    for i, (icon, label, q) in enumerate(welcome_examples):
        with cols[i % 3]:
            if st.button(f"{icon}\n\n{label}", key=f"welcome_{i}", use_container_width=True):
                submit_user_input(q)
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ── Render chat history ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(user_bubble_html(msg["content"]), unsafe_allow_html=True)
    else:
        st.markdown(bot_bubble_html(msg["content"]), unsafe_allow_html=True)


# ── Streaming helper ─────────────────────────────────────────────────────────

def stream_and_display(messages_for_api: list[dict]) -> str:
    """Stream Gemini response with live display, return full text."""
    loading = st.empty()
    loading.markdown(
        '<div class="msg-row bot-row">'
        '<div class="avatar bot-avatar">⚖️</div>'
        '<div class="bot-bubble" style="opacity:0.75;">'
        '<span style="color:#C8A84E;">⚖️ Thinking...</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    placeholder = st.empty()
    full_response = ""
    start = time.time()

    try:
        first_chunk = True
        for chunk in stream_response(API_KEY, SYSTEM_PROMPT, messages_for_api):
            if first_chunk:
                loading.empty()
                first_chunk = False
            full_response += chunk
            placeholder.markdown(
                bot_bubble_html(full_response, streaming=True),
                unsafe_allow_html=True,
            )

        elapsed = time.time() - start
        placeholder.markdown(bot_bubble_html(full_response), unsafe_allow_html=True)
        st.caption(f"⏱️ Response time: {elapsed:.1f}s")
        return full_response

    except Exception as e:
        loading.empty()
        elapsed = time.time() - start
        if full_response:
            placeholder.markdown(bot_bubble_html(full_response), unsafe_allow_html=True)
            st.warning(f"Response may be incomplete ({elapsed:.1f}s). Click 'Continue' to extend.")
            return full_response
        placeholder.empty()
        st.error(f"Failed to get response: {str(e)[:200]}")
        return ""


# ── Stream a pending response (set by previous turn's user input) ───────────
# This block runs AFTER history rendering, so the streamed reply lands at
# the bottom of the chat — no inline duplication, no layout glitches.
if st.session_state.pending_response and st.session_state.messages:
    chat_id = st.session_state.active_chat_id

    # Build API messages from full history
    api_messages = [
        {
            "role": "user",
            "content": SYSTEM_PROMPT + "\n\nUser: " + st.session_state.messages[0]["content"],
        }
    ]
    for m in st.session_state.messages[1:]:
        api_messages.append(m)

    reply = stream_and_display(api_messages)

    if reply:
        st.session_state.messages.append({"role": "assistant", "content": reply})
        if chat_id:
            append_message(chat_id, "assistant", reply)
            # Auto-generate title from first user question
            if len(st.session_state.messages) == 2:
                try:
                    title = generate_title(API_KEY, st.session_state.messages[0]["content"])
                    update_chat_title(chat_id, title)
                except Exception:
                    pass  # title is non-critical

    st.session_state.pending_response = False
    st.rerun()


# ── Action buttons (only when last message is assistant & nothing pending) ──
if (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "assistant"
    and not st.session_state.pending_response
):
    col_regen, col_cont, _ = st.columns([1, 1, 3])
    with col_regen:
        if st.button("🔄 Regenerate", key="regen_btn"):
            # Drop last assistant reply, keep the user msg, queue regeneration
            st.session_state.messages.pop()
            st.session_state.pending_response = True
            st.rerun()
    with col_cont:
        if st.button("➡️ Continue", key="cont_btn"):
            cont_text = "Continue your previous response. Do not repeat what you already said."
            st.session_state.messages.append({"role": "user", "content": cont_text})
            if st.session_state.active_chat_id:
                append_message(st.session_state.active_chat_id, "user", cont_text)
            st.session_state.pending_response = True
            st.rerun()


# ── Chat input ──────────────────────────────────────────────────────────────
prompt = st.chat_input("Ask anything about Indian law...")

if prompt and not st.session_state.pending_response:
    submit_user_input(prompt)
    st.rerun()
