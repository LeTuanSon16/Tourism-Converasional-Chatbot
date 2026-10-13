"""Retrieval-augmented QA over the tourism vector store."""
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from chatbot import config

PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Bạn là trợ lý ảo du lịch Việt Nam. Trả lời bằng tiếng Việt, chỉ dựa trên thông tin sau. "
     "Nếu không biết câu trả lời, hãy nói không biết, đừng tự bịa ra.\n\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL)


@lru_cache
def _retriever():
    if not config.VECTOR_DB_PATH.exists():
        raise FileNotFoundError(f"{config.VECTOR_DB_PATH} not found, run `python -m chatbot.ingest` first.")
    db = FAISS.load_local(str(config.VECTOR_DB_PATH), get_embeddings(), allow_dangerous_deserialization=True)
    return db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": config.TOP_K, "score_threshold": config.SCORE_THRESHOLD},
    )


@lru_cache
def _chain():
    return PROMPT | ChatOpenAI(model=config.LLM_MODEL, temperature=config.TEMPERATURE) | StrOutputParser()


def answer(messages: list[dict]) -> str:
    """Answer the last user message, using earlier messages as conversation history."""
    *history, last = messages
    docs = _retriever().invoke(last["content"])
    return _chain().invoke({
        "context": "\n\n".join(d.page_content for d in docs),
        "history": [(m["role"], m["content"]) for m in history[-config.MAX_HISTORY:]],
        "question": last["content"],
    })
