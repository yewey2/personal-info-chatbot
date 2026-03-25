import streamlit as st
import os
from dotenv import load_dotenv
import traceback

load_dotenv()

st.set_page_config(
    page_title="AMA — Ask Me Anything | Sim Yew Chong",
    page_icon="🤖",
    layout="centered",
)

# --- Sidebar ---
with st.sidebar:
    st.title("About this Bot")
    st.write(
        "This is an AI-powered AMA (Ask Me Anything) assistant for Sim Yew Chong."
    )

    st.subheader("📚 Knowledge Base")
    st.write("✅ Resume (6 chunks loaded)")

    st.divider()

    st.subheader("💡 Suggested Questions")
    suggestions = [
        "What are Yew Chong's technical skills?",
        "Tell me about his experience at CapitaLand",
        "What projects has he worked on?",
        "What is his educational background?",
        "What LLM frameworks does he know?",
    ]
    for suggestion in suggestions:
        if st.button(suggestion, key=suggestion):
            st.session_state["suggested_question"] = suggestion

    st.divider()

    if st.button("🗑️ Clear Conversation"):
        st.session_state.conversation_history = []
        st.session_state.messages = []
        st.rerun()

# --- Session state init ---
if "initialized" not in st.session_state:
    from rag.ingest import load_index, build_index

    index_path = os.path.join(os.path.dirname(__file__), "rag", "faiss_index", "index.faiss")
    if not os.path.exists(index_path):
        with st.spinner("Building FAISS index from resume..."):
            build_index("resume/")
    st.session_state.faiss_index, st.session_state.chunks = load_index()
    st.session_state.conversation_history = []
    st.session_state.messages = []
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": "Hi! I'm Yew Chong's AI assistant. Ask me anything about his background, experience, skills, or projects! 🚀",
            "sources": None,
        }
    )
    st.session_state.initialized = True

# --- Render chat history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources") is not None:
            sources = msg["sources"]
            with st.expander("Sources used"):
                if sources:
                    for s in sources:
                        chunk_id = s["metadata"]["chunk_id"]
                        preview = s["text"][:120] + "..." if len(s["text"]) > 120 else s["text"]
                        st.write(f"**{chunk_id}**: {preview}")
                else:
                    st.write("No specific sources retrieved")

# --- Handle input ---
user_input = st.chat_input("Ask me anything about Yew Chong...")

# Check for suggested question
if "suggested_question" in st.session_state:
    user_input = st.session_state.pop("suggested_question")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input, "sources": None})
    with st.chat_message("user"):
        st.markdown(user_input)

    from agent.chat import is_safe_query, chat
    from rag.retriever import retrieve

    if not is_safe_query(user_input):
        reply = "I'm here to answer questions about Yew Chong's professional background. I can't help with that request! 😊"
        sources = []
    else:
        try:
            context_chunks = retrieve(user_input, top_k=2, score_threshold=0.3)
            reply, st.session_state.conversation_history = chat(
                user_input,
                st.session_state.conversation_history,
                context_chunks,
            )
            sources = context_chunks
        except Exception as e:
            reply = "Something went wrong. Please try again or contact yewchongsim@gmail.com"
            sources = []
            st.error(reply)
            if os.environ.get("DEBUG_MODE", "true").lower() in ["1", "true", "yes"]:
                st.exception(traceback.format_exc())
            

    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(reply)
        with st.expander("Sources used"):
            if sources:
                for s in sources:
                    chunk_id = s["metadata"]["chunk_id"]
                    preview = s["text"][:120] + "..." if len(s["text"]) > 120 else s["text"]
                    st.write(f"**{chunk_id}**: {preview}")
            else:
                st.write("No specific sources retrieved")

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "sources": sources if sources else None}
    )
