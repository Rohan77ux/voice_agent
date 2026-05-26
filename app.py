import streamlit as st
import time
import traceback
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
# ─── Streamlit Render Verification ─────────────────────────────────────────────
try:
    st.write("✅ Streamlit Loaded Successfully")

    # Debug info
    import sys
    import os

    st.sidebar.markdown("### 🔍 Debug Info")
    st.sidebar.write("Python:", sys.version)
    st.sidebar.write("Port:", os.environ.get("PORT", "Not Found"))

except Exception as e:
    st.error(f"Streamlit failed to initialize: {e}")
# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #111118;
    --surface-2: #1a1a25;
    --border: #2a2a3a;
    --accent: #7c3aed;
    --accent-glow: #9f67ff;
    --accent-2: #06b6d4;
    --text: #e8e8f0;
    --text-muted: #7070a0;
    --success: #10b981;
    --danger: #ef4444;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'JetBrains Mono', monospace;
}

.stApp {
    background: var(--bg);
}

.stApp::before {
    content: '';
    position: fixed;
    width: 100%;
    height: 100%;
    background-image:
        linear-gradient(rgba(124,58,237,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(124,58,237,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    z-index: 0;
    pointer-events: none;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800;
    background: linear-gradient(
        135deg,
        #ffffff 0%,
        var(--accent-glow) 50%,
        var(--accent-2) 100%
    );
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-sub {
    color: var(--text-muted);
    font-size: 0.8rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.card-title {
    color: var(--text-muted);
    font-size: 0.75rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    font-weight: 700;
}

.card-content {
    color: var(--text);
    line-height: 1.8;
    font-size: 0.9rem;
}

.stTextInput input {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
}

.stButton button {
    background: linear-gradient(135deg, var(--accent), #5b21b6) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: bold !important;
}

.transcript-box {
    background: var(--surface-2);
    padding: 1rem;
    border-radius: 10px;
    border: 1px solid var(--border);
    max-height: 350px;
    overflow-y: auto;
    white-space: pre-wrap;
}

.chat-container {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 1rem;
    border-radius: 12px;
    margin-bottom: 1rem;
}

.chat-msg {
    margin-bottom: 1rem;
}

.user {
    color: #9f67ff;
    font-weight: bold;
}

.bot {
    color: #06b6d4;
    font-weight: bold;
}

.badge {
    padding: 0.25rem 0.7rem;
    border-radius: 5px;
    font-size: 0.7rem;
    font-weight: bold;
}

.badge-purple {
    background: rgba(124,58,237,0.2);
    color: #b794ff;
}

.badge-green {
    background: rgba(16,185,129,0.15);
    color: #10b981;
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
defaults = {
    "result": None,
    "chat_history": [],
    "pipeline_done": False,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown(
        '<div class="hero-title" style="font-size:1.7rem">🎬 AI Video</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-sub">Meeting Intelligence</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    source = st.text_input(
        "YouTube URL or File Path",
        placeholder="https://youtube.com/..."
    )

    language = st.selectbox(
        "Language",
        ["english", "hinglish"],
        index=0
    )

    run_btn = st.button(
        "⚡ Analyse",
        use_container_width=True
    )

# ─────────────────────────────────────────────────────────────
# MAIN TITLE
# ─────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-title">AI Video Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="hero-sub">Transcribe · Summarise · Chat with Meetings</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ─────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────
if run_btn:

    if not source.strip():
        st.error("Please enter a YouTube URL or file path.")

    else:

        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.pipeline_done = False

        try:

            # AUDIO
            with st.spinner("🔊 Processing audio/video..."):
                chunks = process_input(source)

            # TRANSCRIPT
            with st.spinner("📝 Generating transcript..."):
                transcript = transcribe_all(chunks, language)

            # TITLE
            with st.spinner("🏷️ Generating title..."):
                title = generate_title(transcript)

            # SUMMARY
            with st.spinner("📋 Creating summary..."):
                summary = summarize(transcript)

            # EXTRACTIONS
            with st.spinner("🔍 Extracting insights..."):
                action_items = extract_action_items(transcript)
                decisions = extract_key_decisions(transcript)
                questions = extract_questions(transcript)

            # RAG
            with st.spinner("🧠 Building AI chat engine..."):
                rag_chain = build_rag_chain(transcript)

            # SAVE
            st.session_state.result = {
                "title": title,
                "transcript": transcript,
                "summary": summary,
                "action_items": action_items,
                "key_decisions": decisions,
                "open_questions": questions,
                "rag_chain": rag_chain,
            }

            st.session_state.pipeline_done = True

            st.success("✅ Analysis Complete!")

        except Exception as e:

            st.error(f"❌ Error: {str(e)}")

            st.code(traceback.format_exc())

# ─────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────
if st.session_state.result:

    r = st.session_state.result

    # TITLE
    st.markdown(f"""
    <div class="card">
        <div class="card-title">📌 Session Title</div>
        <div style="font-size:1.5rem;font-weight:700">
            {r['title']}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # SUMMARY + TRANSCRIPT
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">📋 Summary</div>
            <div class="card-content">{r['summary']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        with st.expander("📝 Full Transcript"):
            st.markdown(
                f'<div class="transcript-box">{r["transcript"]}</div>',
                unsafe_allow_html=True
            )

    # ACTIONS / DECISIONS / QUESTIONS
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">✅ Action Items</div>
            <div class="card-content">{r['action_items']}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">🔑 Key Decisions</div>
            <div class="card-content">{r['key_decisions']}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">❓ Open Questions</div>
            <div class="card-content">{r['open_questions']}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # CHAT
    st.markdown("## 💬 Chat with your Meeting")

    if st.session_state.chat_history:

        chat_html = '<div class="chat-container">'

        for msg in st.session_state.chat_history:

            if msg["role"] == "user":

                chat_html += f"""
                <div class="chat-msg">
                    <div class="user">You</div>
                    <div>{msg['content']}</div>
                </div>
                """

            else:

                chat_html += f"""
                <div class="chat-msg">
                    <div class="bot">Assistant</div>
                    <div>{msg['content']}</div>
                </div>
                """

        chat_html += "</div>"

        st.markdown(chat_html, unsafe_allow_html=True)

    user_input = st.text_input(
        "Ask something about the meeting",
        placeholder="What decisions were made?"
    )

    if st.button("Send"):

        if user_input.strip():

            with st.spinner("Thinking..."):

                answer = ask_question(
                    r["rag_chain"],
                    user_input
                )

            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

            st.rerun()

# ─────────────────────────────────────────────────────────────
# EMPTY STATE
# ─────────────────────────────────────────────────────────────
else:

    st.markdown("""
    <div style="
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        padding:5rem 2rem;
        text-align:center;
    ">

        <div style="font-size:4rem">🎬</div>

        <div style="
            font-size:1.6rem;
            font-weight:700;
            margin-top:1rem;
        ">
            Ready to Analyse
        </div>

        <div style="
            color:#888;
            margin-top:0.7rem;
            max-width:500px;
            line-height:1.8;
        ">
            Paste a YouTube URL or local video path,
            choose language,
            and click Analyse.
        </div>

    </div>
    """, unsafe_allow_html=True)