"""Streamlit voice chatbot: `streamlit run app.py`."""
import hashlib

import streamlit as st
from audio_recorder_streamlit import audio_recorder
from streamlit_float import float_init

from chatbot import rag, speech

WELCOME = "Xin chào! Tôi là trợ lý ảo du lịch. Tôi có thể giúp gì cho chuyến đi sắp tới của bạn?"

st.set_page_config(page_title="Tourism Virtual Assistant", page_icon="🤖")
float_init()
st.title("Tourism Virtual Assistant 🤖")
st.session_state.setdefault("messages", [{"role": "assistant", "content": WELCOME}])

footer = st.container()
with footer:
    audio_bytes = audio_recorder(text="", key="recorder")
footer.float("bottom: 0rem;")

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

user_input = st.chat_input("Nhập câu hỏi...")

# The recorder returns its last clip on every rerun: only process new recordings.
if audio_bytes and (digest := hashlib.md5(audio_bytes).hexdigest()) != st.session_state.get("last_audio"):
    st.session_state.last_audio = digest
    with st.spinner("Transcribing..."):
        user_input = speech.transcribe(audio_bytes) or user_input

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking 🤔..."):
            reply = rag.answer(st.session_state.messages)
        st.write(reply)
        if speech.tts_available():
            with st.spinner("Generating audio..."):
                st.audio(speech.synthesize(reply), format="audio/wav", autoplay=True)
    st.session_state.messages.append({"role": "assistant", "content": reply})
