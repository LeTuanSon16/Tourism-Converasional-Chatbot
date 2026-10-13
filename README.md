# Tourism Conversational Chatbot

Vietnamese voice/text chatbot answering travel questions from PDF guides (RAG: FAISS + OpenAI).

```
app.py              Streamlit UI (mic + chat input)
cli.py              Terminal voice chat
chatbot/config.py   Settings (paths, models, retrieval params)
chatbot/ingest.py   PDF -> FAISS index
chatbot/rag.py      Retrieval + answer generation
chatbot/speech.py   Speech-to-text (Google) / text-to-speech (piper)
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env              # set OPENAI_API_KEY (and PIPER_* for voice replies)
python -m chatbot.ingest          # (re)build vectorbase/db_faiss, optional: pass PDF paths
streamlit run app.py
```
