import streamlit as st
import streamlit.components.v1 as components
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
from icons import icon
from animations import inject_animations
from chart_renderer import parse_chart_from_response, truncate_at_chart_tag, render_chart
from rag import retrieve, format_context, index_ready

# ── CONFIG ────────────────────────────────────────────────────────────────────
# API keys: try environment variable first (Render), then Streamlit secrets
if os.environ.get("GEMINI_API_KEYS"):
    import json as _json
    API_KEYS = _json.loads(os.environ["GEMINI_API_KEYS"])
else:
    API_KEYS = list(st.secrets["GEMINI_API_KEYS"])
rotator = init_rotator(API_KEYS)
API_KEY = API_KEYS[0]
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

# ── Animation layer (GSAP + Three.js particles + Anime.js ripple) ───────────
inject_animations()

# ── Base system prompt (no static context — RAG injects per-query) ───────────
_RAG_MODE = "vector" if index_ready() else "BM25 keyword"

BASE_SYSTEM_PROMPT = (
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
    "CHART INSTRUCTIONS:\n"
    "When your response contains numerical comparisons, statistics, punishment tables, "
    "timelines, distributions, or any data that would be clearer as a chart, append a "
    "single <chart>…</chart> block at the very end of your response (after all prose). "
    "Use ONLY one of these JSON structures:\n\n"
    "Bar chart (comparisons, punishments, counts):\n"
    '<chart>{"type":"bar","title":"…","xlabel":"…","ylabel":"…",'
    '"data":[{"label":"…","value":number,"note":"optional text"},…]}</chart>\n\n'
    "Horizontal bar (rankings, long labels):\n"
    '<chart>{"type":"horizontal_bar","title":"…","xlabel":"…","ylabel":"…",'
    '"data":[{"label":"…","value":number},…]}</chart>\n\n'
    "Pie / donut (distributions, categories):\n"
    '<chart>{"type":"pie","title":"…","data":[{"label":"…","value":number},…]}</chart>\n\n'
    "Line chart (trends over time):\n"
    '<chart>{"type":"line","title":"…","xlabel":"…","ylabel":"…",'
    '"data":[{"x":year_or_number,"y":number,"label":"…"},…]}</chart>\n\n'
    "Timeline (chronological events, amendments, history):\n"
    '<chart>{"type":"timeline","title":"…",'
    '"data":[{"year":number,"label":"…"},…]}</chart>\n\n'
    "Rules:\n"
    "- Add a chart ONLY when it genuinely clarifies numeric or comparative information.\n"
    "- Do NOT add a chart for simple factual/definition questions.\n"
    "- The chart block must be valid JSON — no trailing commas, no comments.\n"
    "- Include at least 3 data points in a chart.\n"
)


def build_system_prompt(user_question: str) -> str:
    """Build a system prompt with RAG context retrieved for *user_question*.

    Retrieves the 8 most relevant Q&A pairs from the full 15 K-entry dataset
    and injects them as grounding context.  Falls back gracefully if retrieval
    fails (prompt still works, just without dataset context).
    """
    results = retrieve(user_question, API_KEY, k=8)
    context = format_context(results)
    if context:
        return (
            BASE_SYSTEM_PROMPT
            + "\n\nRelevant Legal Context (retrieved from dataset):\n"
            + context
        )
    return BASE_SYSTEM_PROMPT

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
    """Render a small 'Copy' button that copies the original plain text.

    The icon swap (copy → check on click) runs entirely client-side. The
    SVG markup we feed into the JS template literal contains double quotes
    (xmlns="..." etc.) which would otherwise break the surrounding
    onclick="..." attribute, so we HTML-entity-escape the whole payload —
    the browser unescapes it back to real quotes before executing the JS.
    """
    encoded = urllib.parse.quote(content)
    copy_icon = icon("copy", size=14)
    check_icon = icon("check", size=14)
    js = (
        f"navigator.clipboard.writeText(decodeURIComponent('{encoded}'));"
        f"this.innerHTML=`{check_icon}<span>Copied</span>`;"
        f"setTimeout(()=>{{this.innerHTML=`{copy_icon}<span>Copy</span>`;}},1800);"
    )
    onclick_attr = _html.escape(js, quote=True)
    return (
        f'<button class="copy-btn" type="button" onclick="{onclick_attr}">'
        f'{copy_icon}<span>Copy</span>'
        f'</button>'
    )


def user_bubble_html(content: str) -> str:
    body = md_to_html(content, safe=True)
    return (
        '<div class="msg-row user-row">'
        f'<div class="user-bubble">{body}</div>'
        f'<div class="avatar user-avatar">{icon("user", size=20)}</div>'
        '</div>'
    )


def bot_bubble_html(content: str, streaming: bool = False) -> str:
    # Escape HTML first to prevent any raw <script> from LLM output (defense-in-depth)
    body = md_to_html(content, safe=True) if content else ""
    streaming_cls = " streaming-cursor" if streaming else ""
    copy_html = _copy_button(content) if (content and not streaming) else ""
    return (
        '<div class="msg-row bot-row">'
        f'<div class="avatar bot-avatar">{icon("scale", size=22)}</div>'
        f'<div class="bot-bubble{streaming_cls}">{body}{copy_html}</div>'
        '</div>'
    )


def _snap_js(force: bool) -> str:
    """Shared JS: call __aetherSnap if ready, else scroll last chat element into view.

    Uses scrollTop assignment (not scrollTo) for max cross-browser compat.
    Retries at 150 ms, 400 ms, and 900 ms to survive Streamlit's async DOM updates.
    """
    force_str = "true" if force else "false"
    return f"""<script>
    (function() {{
        var P  = window.parent;
        var PD = P.document;
        function getScroller() {{
            return PD.querySelector('[data-testid="stMainBlockContainer"]') ||
                   PD.querySelector('.main .block-container') ||
                   PD.querySelector('[data-testid="stMain"]');
        }}
        function snap() {{
            if (P.__aetherSnap) {{
                P.__aetherSnap({force_str});
                return;
            }}
            // Fallback: direct scrollTop assignment (works in Safari/Firefox)
            var el = getScroller();
            if (el) el.scrollTop = el.scrollHeight;
        }}
        // Fire immediately, then retry after Streamlit finishes its DOM work
        snap();
        P.setTimeout(snap, 150);
        P.setTimeout(snap, 400);
        P.setTimeout(snap, 900);
    }})();
    </script>"""


def scroll_to_bottom():
    """Snap to bottom after history renders (doesn't reset pause flag)."""
    components.html(_snap_js(force=False), height=0)


def start_scroll_tracker():
    """Force-snap to bottom and clear any pause flag before streaming starts."""
    components.html(_snap_js(force=True), height=0)


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
    st.markdown(
        f'<div class="sidebar-brand">{icon("scale", size=22)}<span>Indian Law Bot</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:rgba(255,255,255,0.7); font-size:0.85rem; margin:0 0 0.6rem 0;'>"
        "Your AI-powered legal assistant</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # New Chat button — Streamlit doesn't accept HTML in button labels, so we
    # use the built-in `icon` parameter to ship a Material Symbols glyph.
    # Material Symbols is the same icon family used by modern SaaS dashboards
    # and renders as a clean, consistent vector across all platforms.
    if st.button("New Chat", key="new_chat", use_container_width=True,
                 icon=":material/add_circle:"):
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
    st.markdown(
        f'<div class="sidebar-section-label">{icon("history", size=14)}'
        f'<span>Chat History</span></div>',
        unsafe_allow_html=True,
    )
    if not all_chats:
        st.markdown(
            "<p style='color:rgba(255,255,255,0.4); font-size:0.8rem;'>No chats yet. Start asking!</p>",
            unsafe_allow_html=True,
        )
    else:
        for chat in all_chats:
            is_active = chat["chat_id"] == st.session_state.active_chat_id
            try:
                dt = datetime.fromisoformat(chat["updated_at"])
                date_str = dt.strftime("%b %d, %I:%M %p")
            except Exception:
                date_str = chat["updated_at"][:16]

            # Each chat row: title button + delete button
            col_btn, col_del = st.columns([5, 1])
            with col_btn:
                if st.button(
                    chat["title"][:30],
                    key=f"chat_{chat['chat_id']}",
                    use_container_width=True,
                    icon=":material/chat_bubble:",
                ):
                    load_chat(chat["chat_id"])
                    st.rerun()
            with col_del:
                if st.button(
                    "",
                    key=f"del_{chat['chat_id']}",
                    icon=":material/delete:",
                    help="Delete this chat",
                ):
                    delete_chat(chat["chat_id"])
                    if st.session_state.active_chat_id == chat["chat_id"]:
                        start_new_chat()
                    st.rerun()

    st.markdown("---")

    # Quick Topics
    st.markdown(
        f'<div class="sidebar-section-label">{icon("bolt", size=14)}'
        f'<span>Quick Topics</span></div>',
        unsafe_allow_html=True,
    )
    # (icon_name, material_icon, label, question)
    topics = [
        ("shield-check", ":material/balance:",          "Fundamental Rights",
         "Explain the Fundamental Rights under the Indian Constitution"),
        ("book",         ":material/menu_book:",         "IPC Sections",
         "What are the most important IPC sections everyone should know?"),
        ("search",       ":material/policy:",            "RTI Act",
         "How do I file an RTI application in India?"),
        ("home",         ":material/home_work:",         "Property Law",
         "What are the key property laws in India for buying and selling?"),
        ("bag",          ":material/verified_user:",     "Consumer Rights",
         "What are my consumer rights under the Consumer Protection Act?"),
        ("briefcase",    ":material/work:",              "Labour Law",
         "What are the major labour laws protecting workers in India?"),
    ]
    for _icon_name, mat_icon, label, question in topics:
        if st.button(label, key=f"topic_{label}", use_container_width=True, icon=mat_icon):
            submit_user_input(question)
            st.rerun()

    st.markdown("---")
    st.markdown(
        '<div class="sidebar-footer">'
        f'<span class="footer-brand">{icon("bolt", size=14)}<span>Powered by Gemini</span></span>'
        '<span class="footer-sub">Aether Neon · Legal AI</span>'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Disclaimer banner (always visible at top) ───────────────────────────────
st.markdown(
    '<div class="disclaimer-banner">'
    f'{icon("alert", size=18)}'
    '<div class="disclaimer-body">'
    '<strong>Disclaimer:</strong> Educational use only. '
    '<strong>Not a substitute</strong> for advice from a qualified advocate.'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ── Hero Banner + Welcome Cards (only when no messages) ─────────────────────
if not st.session_state.messages:
    st.markdown(
        f"""
        <div class="tricolor-bar"></div>
        <div class="hero-banner">
            <h1 class="hero-title">{icon("scale", size=30)}<span>Indian Law Assistant</span></h1>
            <p>Ask me anything about Indian laws, acts, constitutional provisions, or legal rights</p>
        </div>
        <div class="tricolor-bar"></div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<p class="welcome-label">{icon("lightbulb", size=14, cls="icon-glow-cyan")}'
        f'<span style="margin-left:0.4rem;">Try asking about</span></p>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="welcome-state">', unsafe_allow_html=True)

    # (material_icon, label, question)
    welcome_examples = [
        (":material/balance:",       "Fundamental Rights",
         "What are the Fundamental Rights guaranteed by the Indian Constitution?"),
        (":material/menu_book:",     "Article 21",
         "Explain Article 21 of the Indian Constitution and the right to life and liberty."),
        (":material/local_police:",  "How to file an FIR",
         "What is the step-by-step procedure to file an FIR in India?"),
        (":material/handshake:",     "Divorce process",
         "What is the legal process for filing for divorce in India?"),
        (":material/home_work:",     "Property inheritance",
         "What are the property inheritance rights for daughters under Hindu law?"),
        (":material/work_history:",  "Workplace harassment",
         "What legal protections exist against sexual harassment at the workplace in India?"),
    ]

    cols = st.columns(3)
    for i, (mat_icon, label, q) in enumerate(welcome_examples):
        with cols[i % 3]:
            if st.button(label, key=f"welcome_{i}", use_container_width=True, icon=mat_icon):
                submit_user_input(q)
                st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

# ── Render chat history ─────────────────────────────────────────────────────
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(user_bubble_html(msg["content"]), unsafe_allow_html=True)
    else:
        st.markdown(bot_bubble_html(msg["content"]), unsafe_allow_html=True)

# Auto-scroll the page to the latest message after a normal render.
# Skipped while streaming so the placeholder doesn't fight the scroll.
if st.session_state.messages and not st.session_state.pending_response:
    scroll_to_bottom()


# ── Streaming helper ─────────────────────────────────────────────────────────

def stream_and_display(messages_for_api: list[dict]) -> str:
    """Stream Gemini response with live display, return full text."""
    # Single placeholder for both loading state and streaming reply.
    # Using one container avoids the flash/jump between two empty()s.
    placeholder = st.empty()
    placeholder.markdown(
        '<div class="msg-row bot-row thinking-row">'
        f'<div class="avatar bot-avatar">{icon("scale", size=22)}</div>'
        '<div class="bot-bubble" style="opacity:0.92;">'
        '<span class="spinner-ring"></span>'
        '<span class="thinking-text">Analyzing legal context</span>'
        '<span class="typing-dots"><span></span><span></span><span></span></span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    full_response = ""
    start = time.time()

    try:
        for chunk in stream_response(API_KEY, SYSTEM_PROMPT, messages_for_api):
            full_response += chunk
            # Hide the <chart> JSON block while it's being streamed so the
            # user only sees prose — chart is rendered properly at the end.
            display_text = truncate_at_chart_tag(full_response)
            placeholder.markdown(
                bot_bubble_html(display_text, streaming=True),
                unsafe_allow_html=True,
            )

        elapsed = time.time() - start

        # Strip chart block from displayed text and render separately
        clean_text, chart_spec = parse_chart_from_response(full_response)
        placeholder.markdown(bot_bubble_html(clean_text), unsafe_allow_html=True)
        st.markdown(
            f'<div class="response-time">{icon("clock", size=13)}'
            f'<span>Response time: {elapsed:.1f}s</span></div>',
            unsafe_allow_html=True,
        )
        if chart_spec:
            render_chart(chart_spec)

        # Return only clean text so the chart JSON isn't stored in history
        return clean_text

    except Exception as e:
        elapsed = time.time() - start
        if full_response:
            clean_text, chart_spec = parse_chart_from_response(full_response)
            placeholder.markdown(bot_bubble_html(clean_text), unsafe_allow_html=True)
            if chart_spec:
                render_chart(chart_spec)
            st.warning(f"Response may be incomplete ({elapsed:.1f}s). Click 'Continue' to extend.")
            return clean_text
        placeholder.empty()
        st.error(f"Failed to get response: {str(e)[:200]}")
        return ""


# ── Stream a pending response (set by previous turn's user input) ───────────
# This block runs AFTER history rendering, so the streamed reply lands at
# the bottom of the chat — no inline duplication, no layout glitches.
if st.session_state.pending_response and st.session_state.messages:
    chat_id = st.session_state.active_chat_id

    # Start the continuous scroll tracker — it snaps to bottom immediately
    # and then follows page-height growth every 120 ms throughout streaming.
    start_scroll_tracker()

    # Retrieve context relevant to the LATEST user question (not the first),
    # so each turn gets fresh grounding from the full 15 K-entry dataset.
    latest_question = next(
        (m["content"] for m in reversed(st.session_state.messages) if m["role"] == "user"),
        st.session_state.messages[0]["content"],
    )
    system_prompt = build_system_prompt(latest_question)

    # Build API messages from full history
    api_messages = [
        {
            "role": "user",
            "content": system_prompt + "\n\nUser: " + st.session_state.messages[0]["content"],
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
    col_regen, col_cont = st.columns(2)
    with col_regen:
        if st.button("Regenerate", key="regen_btn",
                     use_container_width=True, icon=":material/refresh:"):
            # Drop last assistant reply, keep the user msg, queue regeneration
            st.session_state.messages.pop()
            st.session_state.pending_response = True
            st.rerun()
    with col_cont:
        if st.button("Continue", key="cont_btn",
                     use_container_width=True, icon=":material/arrow_forward:"):
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
